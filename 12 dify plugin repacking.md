````md
# 폐쇄망(Air-gapped) 환경에서 Dify 플러그인 설치 메뉴얼

폐쇄된 업무망(Air-gapped) 환경에서 Dify 플러그인을 설치할 때 발생하는 가장 큰 문제는 **런타임 시점의 종속성(Dependency) 해결**입니다.  
일반적인 `.difypkg` 파일은 메타데이터 위주라 설치 시 외부 PyPI 서버 등에 접속을 시도하기 때문입니다.

시스템 엔지니어 관점에서 가장 **확실하고 재현 가능한 두 가지 솔루션**은 아래와 같습니다.

---

## 방법 1: `dify-plugin-repackaging` 도구 사용 (가장 추천)

인터넷이 되는 환경(Online)에서 모든 종속성 패키지를 미리 포함하여 **오프라인용 패키지**로 재빌드하는 방법입니다.

### 1) 준비 단계 (인터넷 가능 환경)

- **도구 설치:** Dify에서 제공하는 리패키징 툴을 클론합니다.

```bash
git clone https://github.com/langgenius/dify-plugin-repackaging.git
cd dify-plugin-repackaging
````

* **플러그인 다운로드:** 마켓플레이스나 GitHub에서 원하는 플러그인(`.difypkg`)을 받습니다.

### 2) 리패키징 실행

* 아래 명령어로 종속성을 모두 포함한 `-offline.difypkg` 파일을 생성합니다. *(Python 3.12 권장)*

```bash
./plugin_repackaging.sh local ./[대상_플러그인].difypkg
```

* 결과물로 생성된 `xxx-offline.difypkg` 파일을 USB 또는 망간 자료전송 시스템으로 폐쇄망 서버에 반입합니다.

### 3) 폐쇄망 Dify 설정 변경

종속성 설치 시 외부망 접속을 차단하고 로컬 패키지를 우선하도록 환경 변수를 수정해야 합니다.
`docker-compose.yaml` 또는 `.env` 파일에 아래를 추가하세요.

```env
# 서명 확인 비활성화 (로컬 패키지 설치 시 필요할 수 있음)
FORCE_VERIFYING_SIGNATURE=false

# 파일 업로드 용량 제한 상향 (종속성 포함 시 용량이 커짐)
PLUGIN_MAX_PACKAGE_SIZE=524288000
NGINX_CLIENT_MAX_BODY_SIZE=500M
```

---

## 방법 2: Docker 컨테이너 커밋 (설정이 복잡할 경우)

플러그인 개수가 많거나 특정 라이브러리가 복잡하게 얽혀 있다면,
**플러그인이 설치된 상태의 컨테이너를 이미지로 떠서 옮기는 방식**이 가장 확실합니다.

### 절차

1. **외부망:** 폐쇄망과 동일한 버전의 Dify를 설치하고 필요한 모든 플러그인을 정상 설치합니다.
2. **외부망:** `plugin_daemon` 컨테이너가 플러그인과 종속성을 모두 로드했는지 확인합니다.
3. **이미지 추출:**

```bash
docker commit [plugin_daemon_컨테이너ID] my-dify-plugin-daemon:custom
docker save my-dify-plugin-daemon:custom > dify_plugin_image.tar
```

4. **폐쇄망:** 파일을 옮긴 후 이미지를 로드하고 `docker-compose.yaml`에서 `plugin_daemon`의 이미지를 `my-dify-plugin-daemon:custom`으로 변경합니다.

* **주의:** 플러그인 데이터가 저장되는 볼륨(`storage` 등)도 함께 복사해서 옮겨야 설정 정보가 유지됩니다.

---

## 요약 가이드

| 구분        | 방법 1 (패키징)            | 방법 2 (이미지 커밋)          |
| --------- | --------------------- | ---------------------- |
| **장점**    | 관리가 깔끔하고 공식적인 방식      | 종속성 문제를 100% 확실하게 해결   |
| **단점**    | 리패키징 과정에서 빌드 환경 필요    | 이미지 용량이 크고 관리가 무거움     |
| **추천 상황** | 새로운 플러그인을 주기적으로 추가할 때 | 환경 구축 초기나 복잡한 의존성 발생 시 |

---

## 💡 추가 팁

폐쇄망 내부에 **사설 PyPI (Nexus, Artifactory 등)**가 구축되어 있다면,
`.env` 파일에 아래처럼 설정하는 것만으로도 많은 문제가 해결됩니다.

```env
PIP_MIRROR_URL=http://your-internal-pypi/simple
```

---

## 다음 작업 제안

* “리패키징 툴 실행 중 발생하는 에러 로그를 분석해 드릴까요?”
* “Dify 환경 변수 설정을 위한 `docker-compose.yaml` 예시를 만들어 드릴까요?”

```
```
