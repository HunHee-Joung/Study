# N100 환경용 DocLing + Ollama bge-m3 RAG 시스템 설치 가이드

## 🎯 시스템 요구사항

### 최소 요구사항
- **CPU**: Intel N100 (4코어) 또는 동급
- **RAM**: 8GB (16GB 권장)
- **저장공간**: 20GB 여유 공간
- **OS**: Ubuntu 22.04+ / Debian 12+ / CentOS 8+

### 권장 사양
- **RAM**: 16GB+
- **저장공간**: SSD 권장 (HDD는 성능 저하 심각)
- **스왑**: 4GB+ (메모리 부족 시)

## 📦 1단계: 기본 시스템 준비

### Python 환경 설정
```bash
# Python 3.9+ 설치 확인
python3 --version

# pip 업그레이드
python3 -m pip install --upgrade pip

# 가상 환경 생성 (권장)
python3 -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
# rag_env\Scripts\activate     # Windows
```

### 시스템 최적화
```bash
# 스왑 설정 (메모리 8GB 이하인 경우)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# VM 설정 최적화
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 🔧 2단계: Ollama 설치 및 설정

### Ollama 설치
```bash
# Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# 또는 직접 다운로드
# wget https://github.com/ollama/ollama/releases/download/v0.1.17/ollama-linux-amd64
# sudo mv ollama-linux-amd64 /usr/local/bin/ollama
# sudo chmod +x /usr/local/bin/ollama
```

### N100 최적화 환경변수 설정
```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_NUM_PARALLEL=2           # N100 쿼드코어의 절반
export OLLAMA_MAX_LOADED_MODELS=1      # 메모리 절약
export OLLAMA_FLASH_ATTENTION=1        # 메모리 효율성
export OLLAMA_LLM_LIBRARY="cpu"        # CPU 전용 모드

# 적용
source ~/.bashrc
```

### Ollama 서비스 시작
```bash
# 백그라운드 실행
nohup ollama serve > ollama.log 2>&1 &

# 또는 systemd 서비스로 등록
sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=$USER
Environment=OLLAMA_HOST=0.0.0.0:11434
Environment=OLLAMA_NUM_PARALLEL=2
Environment=OLLAMA_MAX_LOADED_MODELS=1
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
```

### bge-m3 모델 설치
```bash
# 모델 다운로드 (약 2.5GB)
ollama pull bge-m3

# 설치 확인
ollama list
```

## 🐍 3단계: Python 패키지 설치

### 핵심 패키지 설치
```bash
# DocLing 설치
pip install docling

# 추가 의존성
pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 기타 필수 패키지
pip install numpy requests psutil aiohttp

# Intel 최적화 (선택사항)
pip install intel-extension-for-pytorch
```

### 설치 확인
```python
# Python에서 실행
import docling
from transformers import AutoTokenizer
import torch
import requests

print("✅ 모든 패키지 설치 완료")
print(f"PyTorch 버전: {torch.__version__}")
print(f"DocLing 사용 가능: {docling.__version__}")
```

## 🚀 4단계: 시스템 테스트

### Ollama 연결 테스트
```bash
# API 테스트
curl http://localhost:11434/api/tags

# 임베딩 테스트
curl -X POST http://localhost:11434/api/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model": "bge-m3", "prompt": "Hello World"}'
```

### Python 통합 테스트
```python
# test_installation.py
import requests
import json

def test_ollama_bge():
    """Ollama bge-m3 테스트"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "bge-m3", "prompt": "테스트 문장"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding = data["embedding"]
            print(f"✅ bge-m3 임베딩 성공: 차원 {len(embedding)}")
            return True
        else:
            print(f"❌ 요청 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        return False

if __name__ == "__main__":
    test_ollama_bge()
```

## 📊 5단계: 성능 모니터링 설정

### 시스템 모니터링 스크립트
```bash
#!/bin/bash
# monitor_n100.sh

echo "=== N100 RAG 시스템 모니터링 ==="
echo "시간: $(date)"
echo ""

# CPU 사용률
echo "🔧 CPU 사용률:"
cat /proc/loadavg

# 메모리 사용량
echo ""
echo "💾 메모리 사용량:"
free -h

# Ollama 프로세스
echo ""
echo "🤖 Ollama 상태:"
ps aux | grep ollama | grep -v grep

# 온도 (lm-sensors 필요)
if command -v sensors &> /dev/null; then
    echo ""
    echo "🌡️ CPU 온도:"
    sensors | grep Core
fi

# 디스크 사용량
echo ""
echo "💿 디스크 사용량:"
df -h / | tail -1

echo ""
echo "==================================="
```

### 자동 모니터링 설정
```bash
# 실행 권한 부여
chmod +x monitor_n100.sh

# cron으로 5분마다 실행
echo "*/5 * * * * /path/to/monitor_n100.sh >> /var/log/n100_monitor.log" | crontab -
```

## 🎮 6단계: RAG 시스템 실행

### 기본 실행
```python
# run_rag.py
from complete_rag_docling_bge import setup_and_run_demo

if __name__ == "__main__":
    setup_and_run_demo()
```

### 서비스로 실행
```bash
# rag_service.py 생성
python3 rag_service.py &

# 프로세스 확인
ps aux | grep rag_service
```

## 🔍 문제 해결 가이드

### 일반적인 문제들

#### 1. Ollama 연결 실패
```bash
# 서비스 상태 확인
systemctl status ollama

# 로그 확인
journalctl -u ollama -f

# 포트 확인
netstat -tlnp | grep 11434
```

#### 2. 메모리 부족
```bash
# 스왑 사용량 확인
swapon --show

# 프로세스 메모리 사용량
ps aux --sort=-%mem | head -10

# 메모리 정리
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

#### 3. 성능 저하
```bash
# CPU 주파수 확인
cat /proc/cpuinfo | grep MHz

# 열적 스로틀링 확인
dmesg | grep -i thermal

# I/O 대기 확인
iostat 1 5
```

#### 4. 모델 로딩 실패
```bash
# 모델 재설치
ollama rm bge-m3
ollama pull bge-m3

# 캐시 정리
rm -rf ~/.ollama/models/blobs/*
```

## ⚡ N100 특화 최적화 팁

### 1. CPU 최적화
```bash
# CPU 거버너 설정
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 인터럽트 분산
echo 2 | sudo tee /proc/irq/*/smp_affinity
```

### 2. 메모리 최적화
```python
# Python 메모리 최적화
import gc
import os

# 가비지 컬렉션 강화
gc.set_threshold(700, 10, 10)

# 메모리 매핑 최적화
os.environ['MALLOC_MMAP_THRESHOLD_'] = '65536'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'
```

### 3. 네트워크 최적화
```bash
# TCP 최적화
echo 'net.core.rmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
```

## 📈 성능 벤치마크

### 예상 성능 지표 (N100 기준)
- **문서 처리**: 1-3 페이지/분
- **청킹 속도**: 100-300 청크/분
- **임베딩 생성**: 5-15 청크/분
- **검색 지연시간**: 100-500ms
- **메모리 사용량**: 2-6GB
- **CPU 사용률**: 60-90% (피크 시)

### 벤치마크 실행
```python
# benchmark.py
from complete_rag_docling_bge import *
import time

def run_benchmark():
    config = RAGConfig(batch_size=1)
    rag = N100OptimizedRAG(config)
    
    # 문서 추가 성능 측정
    start = time.time()
    rag.add_document_from_text("test", "테스트 문서" * 100)
    print(f"문서 추가 시간: {time.time() - start:.2f}초")
    
    # 검색 성능 측정
    start = time.time()
    results = rag.search("테스트")
    print(f"검색 시간: {time.time() - start:.2f}초")

if __name__ == "__main__":
    run_benchmark()
```

## 🎯 다음 단계

1. **프로덕션 배포**: Docker 컨테이너 또는 systemd 서비스로 안정적 운영
2. **스케일링**: 다중 인스턴스 또는 로드 밸런싱 고려
3. **모니터링**: Prometheus + Grafana로 메트릭 수집
4. **백업**: 임베딩 데이터베이스 정기 백업 전략
5. **업데이트**: 모델 및 라이브러리 버전 관리

## 🆘 지원 및 커뮤니티

- **DocLing**: https://github.com/docling-project/docling
- **Ollama**: https://github.com/ollama/ollama
- **bge-m3**: https://huggingface.co/BAAI/bge-m3
- **이슈 리포트**: GitHub Issues 또는 커뮤니티 포럼

---

✅ **설치 완료 체크리스트**
- [ ] Python 3.9+ 설치 및 가상환경 설정
- [ ] Ollama 설치 및 서비스 등록
- [ ] bge-m3 모델 다운로드
- [ ] DocLing 및 의존성 패키지 설치
- [ ] 시스템 최적화 설정 적용
- [ ] 연결 테스트 성공
- [ ] 데모 실행 성공
- [ ] 모니터링 시스템 설정

🎉 **축하합니다! N100 환경용 RAG 시스템이 준비되었습니다.**
