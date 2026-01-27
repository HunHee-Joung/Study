# Dify Ubuntu → Red Hat 이관(오프라인) 요약 보고서

## 1. 목적
Ubuntu 환경에서 동작 중인 **Dify(v1.11.3)**를 인터넷 연결이 제한된(폐쇄망/내부망) **Red Hat 서버**로 이관하여, 동일 버전·동일 컨테이너 이미지 기반으로 서비스를 재구성하고 **기존 LLM 서버(Ollama/vLLM)**까지 연동한다.

---

## 2. 범위 및 전제
- **대상 버전**: Dify `v1.11.3`
- **이관 방식**: 소스 코드(Repo) + 컨테이너 이미지(Offline tar) 패키징 후 전송
- **런타임 선택지**
  - **Docker 기반**: Red Hat에 Docker 설치 가능/허용 시
  - **Podman + podman-compose 기반**: Docker 미설치/제한 환경 고려

---

## 3. Ubuntu 측 작업(패키지 준비)

### 3.1 Dify 소스 확보
- GitHub에서 Dify를 클론 후 `v1.11.3`로 체크아웃

### 3.2 필수 이미지 사전 다운로드(Pull)
- **Dify 핵심 이미지**
  - `langgenius/dify-api:1.11.3`
  - `langgenius/dify-web:1.11.3`
  - `langgenius/dify-sandbox:0.2.10`
  - `langgenius/dify-plugin-daemon:0.5.2-local`
- **인프라 이미지**
  - `postgres:15-alpine`
  - `redis:6-alpine`
  - `nginx:latest`
  - `semitechnologies/weaviate:1.19.0`

### 3.3 이미지 저장 및 오프라인 패키지 생성
- `docker save`로 이미지들을 `dify-images.tar`로 저장
- 소스 폴더(`dify/`) + 이미지 tar를 묶어 `dify-v1.11.3-offline.tar.gz` 생성

**산출물(전송 파일)**
- `dify-v1.11.3-offline.tar.gz` (소스 + 이미지 번들)

---

## 4. Red Hat 측 설치/기동 절차

### 4.1 압축 해제 및 이미지 로드
- 홈 디렉토리에서 압축 해제
- (Docker 사용 시) `docker load -i dify-images.tar`

### 4.2 환경설정(.env)
- `dify/docker` 경로에서 `.env.example` → `.env` 복사 후 값 수정
- **핵심 설정 항목**
  - 서비스 URL  
    - `CONSOLE_WEB_URL`, `SERVICE_API_URL`, `APP_WEB_URL` = `http://<RedHat 서버 IP>`
  - 보안 키
    - `SECRET_KEY=<강한 임의값>`
  - DB 접속
    - `DB_USERNAME`, `DB_PASSWORD`, `DB_HOST=db`, `DB_PORT=5432`, `DB_DATABASE=dify`

> 참고: 문서 내 `~/dify-package/...`와 `/opt/dify-package/...` 경로가 혼재될 수 있으므로, 운영 시 **한 경로로 통일** 권장

### 4.3 기동 및 점검
- (Docker) `docker compose up -d`
- 상태/로그 확인
  - `docker compose ps`
  - `docker compose logs -f`
- 이미지 로드 확인
  - `docker images | grep dify`

---

## 5. 기존 LLM 서버 연동(Ollama / vLLM)

### 5.1 Dify 콘솔에서 모델 제공자 추가
- **Ollama**
  - Provider: Ollama
  - Base URL: `http://<Ollama서버IP>:11434`
  - Model: 예) `qwen2.5:7b`
- **vLLM(OpenAI 호환)**
  - Provider: OpenAI-API-compatible
  - Base URL: `http://<vLLM서버IP>:8000/v1`
  - API Key: `dummy`
  - Model: vLLM에 로드된 모델명

---

## 6. 연결 테스트(네트워크/헬스체크)
Red Hat 서버에서 아래 호출로 통신 여부를 확인한다.

- Ollama
  - `curl http://<Ollama서버IP>:11434/api/tags`
- vLLM
  - `curl http://<vLLM서버IP>:8000/v1/models`
- Dify 헬스체크(로컬)
  - `curl http://localhost/health`

---

## 7. 폐쇄망(오프라인) 플러그인 고려사항
Dify v1.x부터 플러그인 시스템 도입으로 **마켓플레이스 접근 불가 환경**에서 문제가 발생할 수 있다.

- 필요 시 `.env`에 추가
  - `MARKETPLACE_ENABLED=false`

---

## 8. Podman 기반 대안(도커 미설치 환경)

### 8.1 Podman 설치
- `sudo dnf install -y podman`

### 8.2 podman-compose 오프라인 설치(ubuntu에서 wheel 다운로드 후 전송)
- Ubuntu에서
  - `pip3 download podman-compose`
  - 다운로드 결과를 tar로 묶어 전송
- Red Hat에서
  - `pip3 install --user --no-index --find-links=. podman-compose`
  - `podman-compose --version` 확인

### 8.3 이미지 로드/기동
- `podman load -i dify-images-complete.tar`
- `podman-compose up -d`

---

## 9. 자주 발생하는 이슈 및 대응(중요)

### 9.1 “누락 이미지” 문제
`docker compose config --images` 결과에 따라, 실제로는 `busybox`, `ubuntu/squid` 등 **추가 이미지**가 필요할 수 있다.

**권장 해결 절차(Ubuntu에서 완전 번들 재생성)**
1. `cd ~/dify-package/dify/docker`
2. `docker compose pull` (필요 이미지 전부 pull)
3. `docker save $(docker compose config --images) -o dify-images-complete.tar`
4. Red Hat으로 전송 후 로드
   - Docker: `docker load -i dify-images-complete.tar`
   - Podman: `podman load -i dify-images-complete.tar`

---

## 10. 최종 결과물 체크리스트
- [ ] Red Hat 서버에서 Dify 컨테이너 정상 기동(`compose ps`)
- [ ] Dify 콘솔 접속 가능(설정한 `CONSOLE_WEB_URL`)
- [ ] `/health` 응답 정상
- [ ] Ollama/vLLM 모델 목록 조회 정상(`api/tags`, `/v1/models`)
- [ ] 폐쇄망 환경에서 플러그인 관련 오류 시 `MARKETPLACE_ENABLED=false` 적용
