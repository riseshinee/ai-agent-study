# AI Agent Study

이 프로젝트는 OREILLY의 **AI 에이전트 엔지니어링** 책을 보면서 실습하고 학습한 내용을 정리하는 저장소입니다.

## 목적

- 책의 예제를 직접 구현하며 AI 에이전트 개발 흐름 익히기
- 실습 코드와 실험 결과를 단계별로 정리하기
- 학습 과정에서 얻은 인사이트를 기록하기

## 실행하기
- **예제 코드를 실행하려면 유료 OpenAI API(예: GPT-4, GPT-3.5 Turbo 등) 접근 권한이 필요합니다. 무료 계정은 실행이 불가하거나 제한될 수 있습니다.**
- OpenAI API 키 발급 요약: [OpenAI 홈페이지](https://openai.com/ko-KR)에 접속 → 로그인 → 우측 상단 'API' 클릭 → 'API Keys'에서 새 키 생성
- Python 설치 (Homebrew 기준):  
  ```
  brew install python
  ```
- uv 설치 및 의존성 패키지 설치 (pyproject.toml 필요, /.venv 디렉토리에 가상환경을 생성하고, uv.lock에 정의된 패키지들을 설치):
  ```
  brew install uv
  uv pip install -r pyproject.toml
  ```
  
- 가상환경 활성화:
  ```
  source .venv/bin/activate
  ```