# DGX Spark에서 OpenCode + Ollama + Qwen3-Coder 30B 바이브코딩 환경 구축 가이드

> **대상 환경**: NVIDIA DGX Spark (128GB 통합메모리)  
> **목표**: 온프레미스 완전 로컬 AI 코딩 에이전트 구축  
> **소요 시간**: 약 30~40분

---

## 전체 흐름 요약

```
[STEP 1] Ollama 설치
    ↓
[STEP 2] Qwen3-Coder 30B 모델 다운로드
    ↓
[STEP 3] 컨텍스트 윈도우 확장 (필수!)
    ↓
[STEP 4] Ollama 서버 실행 확인
    ↓
[STEP 5] Node.js 설치 (OpenCode 필수 의존성)
    ↓
[STEP 6] OpenCode 설치
    ↓
[STEP 7] OpenCode 설정 파일 작성 (Ollama + Qwen3-Coder 연결)
    ↓
[STEP 8] 프로젝트 폴더에서 OpenCode 실행
    ↓
[STEP 9] 바이브코딩 시작!
```

---

## STEP 1. Ollama 설치

DGX Spark는 Ubuntu(DGX OS) 기반이므로 Linux 설치 명령어를 사용합니다.

```bash
# Ollama 설치 (한 줄이면 끝)
curl -fsSL https://ollama.ai/install.sh | sh
```

설치 확인:
```bash
ollama --version
# 출력 예: ollama version 0.x.x
```

Ollama 서비스가 자동으로 시작됩니다. 확인:
```bash
systemctl status ollama
# Active: active (running) 이면 정상
```

만약 서비스가 안 떠있으면:
```bash
sudo systemctl start ollama
sudo systemctl enable ollama   # 부팅 시 자동 시작
```

---

## STEP 2. Qwen3-Coder 30B 모델 다운로드

```bash
# 모델 다운로드 (약 19GB, 네트워크 속도에 따라 5~20분 소요)
ollama pull qwen3-coder:30b
```

다운로드 완료 확인:
```bash
ollama list
# NAME                    SIZE      
# qwen3-coder:30b         19GB
```

> **참고**: DGX Spark의 4TB NVMe SSD에 저장되므로 용량 걱정 없습니다.

---

## STEP 3. 컨텍스트 윈도우 확장 (⚠️ 매우 중요!)

**이 단계를 빠뜨리면 OpenCode에서 tool calling이 작동하지 않습니다.**

Ollama는 기본 컨텍스트 윈도우가 4096 토큰입니다.
OpenCode의 에이전틱 기능(파일 읽기/쓰기, bash 실행 등)을 사용하려면
최소 16K, 권장 32K 이상이 필요합니다.

DGX Spark는 128GB 메모리이므로 넉넉하게 64K로 설정합니다.

```bash
# 모델을 대화형으로 실행
ollama run qwen3-coder:30b

# Ollama 프롬프트 안에서 아래 명령어를 순서대로 입력:
>>> /set parameter num_ctx 65536
Set parameter 'num_ctx' to '65536'

>>> /save qwen3-coder:30b-64k
Created new model 'qwen3-coder:30b-64k'

>>> /bye
```

확인:
```bash
ollama list
# NAME                        SIZE      
# qwen3-coder:30b             19GB
# qwen3-coder:30b-64k         19GB    ← 새로 생성된 모델
```

> **왜 64K인가?**
> - Qwen3-Coder 30B는 네이티브 256K까지 지원하지만
> - 컨텍스트가 커질수록 메모리 사용량과 응답 속도에 영향
> - 64K는 바이브코딩에 충분하면서 속도도 쾌적한 밸런스 포인트
> - DGX Spark 128GB에서 모델(19GB) + 64K KV캐시(~10GB) = 약 30GB 사용

---

## STEP 4. Ollama 서버 실행 확인

OpenCode는 Ollama의 OpenAI 호환 API 엔드포인트로 통신합니다.

```bash
# Ollama API가 정상 동작하는지 확인
curl http://localhost:11434/v1/models
```

정상이면 설치된 모델 목록이 JSON으로 출력됩니다.

간단한 테스트:
```bash
curl http://localhost:11434/api/chat \
  -d '{
    "model": "qwen3-coder:30b-64k",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

응답이 오면 Ollama 준비 완료입니다.

---

## STEP 5. Node.js 설치

OpenCode는 JavaScript/TypeScript 기반이라 Node.js 20 이상이 필요합니다.

```bash
# Node.js 버전 확인 (이미 설치되어 있을 수 있음)
node --version

# 없거나 버전이 낮으면 설치
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# 설치 확인
node --version   # v22.x.x
npm --version    # 10.x.x
```

---

## STEP 6. OpenCode 설치

```bash
# 방법 1: 공식 설치 스크립트 (권장)
curl -fsSL https://opencode.ai/install | bash

# 설치 후 PATH에 추가 (스크립트가 안내해줌)
export PATH="$HOME/.opencode/bin:$PATH"

# 영구 적용
echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

또는:
```bash
# 방법 2: npm으로 설치
npm install -g opencode
```

설치 확인:
```bash
opencode --version
```

---

## STEP 7. OpenCode 설정 파일 작성 (핵심!)

OpenCode가 Ollama의 Qwen3-Coder 모델을 사용하도록 설정합니다.

```bash
# 설정 디렉토리 생성
mkdir -p ~/.config/opencode

# 설정 파일 작성
cat > ~/.config/opencode/opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (Local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen3-coder:30b-64k": {
          "name": "Qwen3 Coder 30B (64K context)",
          "tools": true
        }
      }
    }
  }
}
EOF
```

> **설정 파일 핵심 포인트:**
> - `baseURL`: Ollama의 OpenAI 호환 엔드포인트 (반드시 /v1 포함)
> - `models` 키: STEP 3에서 만든 모델명과 정확히 일치해야 함
> - `"tools": true`: 이걸 넣어야 파일 생성/수정, bash 실행 등 에이전틱 기능 활성화

---

## STEP 8. 프로젝트 폴더에서 OpenCode 실행

```bash
# 프로젝트 폴더 생성 (또는 기존 프로젝트로 이동)
mkdir -p ~/my-project
cd ~/my-project

# OpenCode 실행
opencode
```

OpenCode TUI(터미널 UI)가 열리면:

```
█▀▀█ █▀▀█ █▀▀ █▀▀▄ █▀▀ █▀▀█ █▀▀▄ █▀▀
█░░█ █░░█ █▀▀ █░░█ █░░ █░░█ █░░█ █▀▀
▀▀▀▀ █▀▀▀ ▀▀▀ ▀  ▀ ▀▀▀ ▀▀▀▀ ▀▀▀  ▀▀▀
```

### 첫 실행 시 할 일:

```
1) /init 입력 → 프로젝트 초기화 (agents.mmd 파일 생성)

2) /connect 입력 → 프로바이더 연결
   → "ollama" 검색하여 선택

3) /models 입력 → 모델 선택
   → "Qwen3 Coder 30B (64K context)" 선택
```

### 연결 테스트:

프롬프트에 간단한 질문을 입력합니다:
```
> Python으로 Hello World 출력하는 코드 만들어줘
```

AI가 파일을 생성하고 코드를 작성하면 성공입니다!

---

## STEP 9. 바이브코딩 시작!

### OpenCode 주요 모드

| 모드 | 설명 | 전환 방법 |
|------|------|-----------|
| **Build** | 실제로 파일을 생성/수정/삭제 (기본 모드) | 기본값 |
| **Plan** | 파일 수정 없이 계획만 세움 | Tab 키 |

### 바이브코딩 예시 대화

```
> FastAPI로 사용자 CRUD API 만들어줘. 
  SQLite 사용하고, Pydantic 모델도 포함해줘.

(AI가 자동으로 파일 생성)
✓ Created main.py
✓ Created models.py  
✓ Created database.py
✓ Created requirements.txt

> requirements.txt 기반으로 의존성 설치해줘

(AI가 bash 도구를 사용해 pip install 실행)
✓ Executed: pip install -r requirements.txt

> 서버 실행해서 테스트해봐

(AI가 uvicorn 실행)
✓ Executed: uvicorn main:app --reload
```

### 유용한 명령어

```
/init          프로젝트 초기화
/connect       프로바이더 연결/변경
/models        모델 선택/변경
/compact       컨텍스트 정리 (대화가 길어졌을 때)
/clear         대화 초기화
/cost          토큰 사용량 확인
Tab            Plan ↔ Build 모드 전환
@파일명         특정 파일을 컨텍스트에 추가
```

---

## (선택) GPT-OSS 120B도 함께 설정하기

DGX Spark 128GB이면 두 모델 모두 설정해두고 필요에 따라 전환 가능합니다.

```bash
# GPT-OSS 120B 다운로드 (약 60GB, 시간 소요)
ollama pull gpt-oss:120b

# 컨텍스트 확장
ollama run gpt-oss:120b
>>> /set parameter num_ctx 32768
>>> /save gpt-oss:120b-32k
>>> /bye
```

설정 파일에 모델 추가:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (Local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen3-coder:30b-64k": {
          "name": "Qwen3 Coder 30B (64K) - 빠른 코딩용",
          "tools": true
        },
        "gpt-oss:120b-32k": {
          "name": "GPT-OSS 120B (32K) - 깊은 추론용",
          "tools": true
        }
      }
    }
  }
}
```

OpenCode 안에서 `/models` 명령으로 언제든 전환 가능합니다.

---

## 트러블슈팅

### Tool calling이 안 될 때
```bash
# 원인 99%: 컨텍스트 윈도우가 너무 작음
# STEP 3을 다시 확인하고, 최소 16384 이상으로 설정

ollama run qwen3-coder:30b-64k
>>> /show parameter num_ctx
# 65536이 출력되어야 정상
```

### Ollama 연결 실패
```bash
# Ollama 서비스 상태 확인
systemctl status ollama

# 안 되면 재시작
sudo systemctl restart ollama

# API 포트 확인
curl http://localhost:11434/api/tags
```

### OpenCode에서 모델이 안 보일 때
```bash
# 설정 파일 경로 확인
cat ~/.config/opencode/opencode.json

# 모델명이 ollama list 출력과 정확히 일치하는지 확인
ollama list
```

### 응답이 너무 느릴 때
```bash
# GPU 사용 여부 확인
nvidia-smi
# Ollama 프로세스가 GPU 메모리를 사용하고 있어야 함

# 컨텍스트 크기 줄이기 (속도 ↔ 컨텍스트 트레이드오프)
# 64K → 32K로 줄이면 속도 향상
ollama run qwen3-coder:30b-64k
>>> /set parameter num_ctx 32768
>>> /save qwen3-coder:30b-32k
>>> /bye
```

---

## DGX Spark에서의 예상 성능

| 항목 | 예상 수치 |
|------|-----------|
| 모델 로딩 메모리 | ~19GB / 128GB |
| 64K 컨텍스트 시 총 메모리 | ~30GB / 128GB |
| Decode 속도 (예상) | 30~50 tok/s |
| 짧은 코드 생성 (함수 1개) | 3~5초 |
| 파일 1개 전체 생성 | 15~30초 |
| 바이브코딩 체감 | 쾌적 ✅ |

---

*작성일: 2026-02-17*
*환경: NVIDIA DGX Spark (128GB) + Ubuntu (DGX OS)*
