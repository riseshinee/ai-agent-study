# fine_tune_function_calling.py
"""LoRA를 사용하여 함수 호출을 위한 LLM 파인튜닝을 수행하는 깔끔하고 모듈화된 스크립트입니다.

이 스크립트는 Bonus Unit 노트북의 단계들을 하나의 Python 진입점으로 통합합니다.
기본 사용법:

    HF_TOKEN=<your_token> python fine_tune_function_calling.py \
        --model google/gemma-2-2b-it \
        --dataset Jofthomas/hermes-function-calling-thinking-V1 \
        --output_dir gemma-2-2B-function-call-ft

모든 옵션을 보려면 `--help`를 참조하세요.
"""
from __future__ import annotations

import argparse
import os
import platform
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

###############################################################################
# 특수 토큰 및 채팅 템플릿 헬퍼
###############################################################################

class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_response>"
    eotool_response = "</tool_response>"
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def list(cls) -> List[str]:
        return [c.value for c in cls]

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content']|trim + '<end_of_turn><eos>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)

###############################################################################
# 데이터셋 전처리
###############################################################################

def _merge_system_into_first_user(messages: List[Dict[str, str]]) -> None:
    """선행 시스템 메시지를 후속 사용자 메시지에 병합합니다."""
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        messages.pop(0)
        if not messages or messages[0]["role"] != "human":
            raise ValueError("시스템 메시지 다음에 사용자 메시지가 있어야 합니다.")
        messages[0][
            "content"
        ] = (
            f"{system_content}또한, 함수를 호출하기 전에 시간을 갖고 "
            "호출할 함수를 계획하세요. 생각하는 과정을 "
            "<think>{your thoughts}</think> 사이에 작성하세요.\n\n" + messages[0]["content"]
        )


def build_preprocess_fn(tokenizer):
    """원시 샘플을 토크나이즈된 프롬프트로 매핑하는 함수를 반환합니다."""
    def _preprocess(sample):
        messages = sample["messages"].copy()
        _merge_system_into_first_user(messages)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": prompt}

    return _preprocess


def load_and_prepare_dataset(ds_name: str, tokenizer, max_train: int, max_eval: int) -> DatasetDict:
    """데이터셋을 로드하고 전처리 및 학습/테스트 분할을 적용합니다."""
    raw = load_dataset(ds_name).rename_column("conversations", "messages")
    processed = raw.map(build_preprocess_fn(tokenizer), remove_columns="messages")
    split = processed["train"].train_test_split(test_size=0.1, seed=42)
    split["train"] = split["train"].select(range(max_train))
    split["test"] = split["test"].select(range(max_eval))
    return split

###############################################################################
# 모델 및 토크나이저 헬퍼
###############################################################################

def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list(),
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def build_model(model_name: str, tokenizer, load_4bit: bool = False):
    """모델 로드. Mac에서는 bitsandbytes가 불안정하므로 4bit 양자화 비활성화."""
    kwargs = {
        "attn_implementation": "eager",  # flash_attn 호환성 (Phi-3 window_size 등)
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    # load_4bit=True이고 Mac이 아닐 때만 4bit 양자화 사용 (bitsandbytes는 Mac에서 불안정)
    use_4bit = load_4bit and platform.system() != "Darwin"
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    return model

###############################################################################
# PEFT / LoRA 헬퍼
###############################################################################

def build_lora_config(r: int = 16, alpha: int = 64, dropout: float = 0.05) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "gate_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "lm_head",
            "embed_tokens",
        ],
        task_type=TaskType.CAUSAL_LM,
    )

###############################################################################
# 학습
###############################################################################

def train(
    model,
    tokenizer,
    dataset: DatasetDict,
    peft_cfg: LoraConfig,
    output_dir: str,
    epochs: int = 1,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    max_seq_len: int = 1500,
):
    train_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=5,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to=None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    return trainer

###############################################################################
# 명령줄 인터페이스(CLI)
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA를 사용하여 함수 호출을 위한 LLM 파인튜닝을 수행합니다.")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="기본 모델 이름 또는 경로")
    parser.add_argument("--dataset", default="Jofthomas/hermes-function-calling-thinking-V1", help="HuggingFace 데이터셋")
    parser.add_argument("--output_dir", default="ch07/fine_tuned_model/gemma-2-2B-function-call-ft", help="체크포인트 저장 경로")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_train", type=int, default=100, help="빠른 실행을 위한 학습 데이터 행 수")
    parser.add_argument("--max_eval", type=int, default=10, help="빠른 실행을 위한 평가 데이터 행 수")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_username", default=None, help="모델 푸시를 위한 HuggingFace 사용자명")
    parser.add_argument("--load_4bit", action="store_true", help="4비트 양자화 모드로 기본 모델 로드")
    return parser.parse_args()


def maybe_push_to_hub(trainer: SFTTrainer, tokenizer, username: str, output_dir: str):
    if not username:
        print("HuggingFace 사용자명이 제공되지 않아 push_to_hub을 건너뜁니다.")
        return
    repo = f"{username}/{Path(output_dir).name}"
    print(f"\n어댑터 및 토크나이저를 https://huggingface.co/{repo} 에 푸시하는 중 …")
    trainer.push_to_hub(repo)
    tokenizer.push_to_hub(repo, token=os.environ.get("HF_TOKEN"))

###############################################################################
# 진입점
###############################################################################

def main():
    args = parse_args()

    tokenizer = build_tokenizer(args.model)
    model = build_model(args.model, tokenizer, load_4bit=args.load_4bit)

    dataset = load_and_prepare_dataset(
        args.dataset, tokenizer, max_train=args.max_train, max_eval=args.max_eval
    )

    lora_cfg = build_lora_config()
    results = train(
        model,
        tokenizer,
        dataset,
        lora_cfg,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )

    print("\n학습 완료! 🎉")
    print(results)


if __name__ == "__main__":
    main()