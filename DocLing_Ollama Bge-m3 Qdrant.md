# Qdrant 벡터 데이터베이스 설치 및 설정 가이드

## 🎯 Qdrant란?

Qdrant는 고성능 벡터 데이터베이스로, 대규모 임베딩 벡터의 저장과 유사도 검색에 최적화되어 있습니다.

### 주요 특징
- **빠른 검색**: HNSW 알고리즘으로 밀리초 단위 검색
- **스케일링**: 수십억 개 벡터까지 처리 가능  
- **필터링**: 메타데이터 기반 정교한 필터링
- **RESTful API**: 언어에 관계없이 쉬운 통합
- **N100 최적화**: 저사양 CPU에서도 우수한 성능

## 🐳 1단계: Docker로 Qdrant 설치 (권장)

### Docker 설치 확인
```bash
# Docker 버전 확인
docker --version

# Docker 없으면 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

### Qdrant 컨테이너 실행
```bash
# 기본 실행 (메모리 저장)
docker run -p 6333:6333 qdrant/qdrant

# 데이터 영구 저장 (권장)
mkdir -p ~/qdrant_data
docker run -p 6333:6333 \
    -v ~/qdrant_data:/qdrant/storage \
    qdrant/qdrant

# N100 최적화 설정으로 실행
docker run -p 6333:6333 \
    -v ~/qdrant_data:/qdrant/storage \
    -e QDRANT__SERVICE__HTTP_PORT=6333 \
    -e QDRANT__SERVICE__GRPC_PORT=6334 \
    -e QDRANT__STORAGE__HNSW_CONFIG__EF_CONSTRUCT=100 \
    -e QDRANT__STORAGE__HNSW_CONFIG__M=16 \
    --memory=2g \
    --cpus=2 \
    qdrant/qdrant
```

### Docker Compose 사용 (영구 서비스)
```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant
    container_name: qdrant_n100
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
      # N100 최적화 설정
      - QDRANT__STORAGE__HNSW_CONFIG__EF_CONSTRUCT=100
      - QDRANT__STORAGE__HNSW_CONFIG__M=16
      - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=2
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
    restart: unless-stopped
```

```bash
# 서비스 시작
docker-compose up -d

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs qdrant
```

## 🖥️ 2단계: 네이티브 설치 (선택사항)

### Linux (Ubuntu/Debian)
```bash
# Qdrant 바이너리 다운로드
wget https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
sudo mv qdrant /usr/local/bin/

# 설정 파일 생성
sudo mkdir -p /etc/qdrant
sudo tee /etc/qdrant/config.yaml > /dev/null <<EOF
service:
  http_port: 6333
  grpc_port: 6334

storage:
  storage_path: "/var/lib/qdrant"
  hnsw_config:
    ef_construct: 100
    m: 16
  performance:
    max_search_threads: 2
EOF

# 데이터 디렉토리 생성
sudo mkdir -p /var/lib/qdrant
sudo chown $USER:$USER /var/lib/qdrant

# systemd 서비스 생성
sudo tee /etc/systemd/system/qdrant.service > /dev/null <<EOF
[Unit]
Description=Qdrant Vector Database
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/local/bin/qdrant --config-path /etc/qdrant/config.yaml
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl enable qdrant
sudo systemctl start qdrant
```

## 🔧 3단계: N100 최적화 설정

### Qdrant 성능 튜닝
```yaml
# config.yaml 최적화 설정
service:
  http_port: 6333
  grpc_port: 6334
  max_request_size_mb: 32

storage:
  storage_path: "./storage"
  
  # HNSW 최적화 (정확도 vs 속도)
  hnsw_config:
    ef_construct: 100    # 낮추면 빠르지만 정확도 감소 (기본값: 200)
    m: 16               # 연결 수, N100에서는 16 권장 (기본값: 16)
    
  # 성능 최적화
  performance:
    max_search_threads: 2           # N100 쿼드코어의 절반
    max_optimization_threads: 1     # 백그라운드 최적화
    
  # 메모리 최적화
  wal_config:
    wal_capacity_mb: 32
    wal_segments_ahead: 0
    
  # 동시성 제한
  optimizer_config:
    max_segment_size_kb: 200000     # 200MB 세그먼트
    memmap_threshold_kb: 200000
    indexing_threshold_kb: 20000
    flush_interval_sec: 5

# 클러스터 설정 (단일 노드)
cluster:
  enabled: false

# 로그 레벨
log_level: INFO
```

### 시스템 수준 최적화
```bash
# 파일 디스크립터 증가
echo 'fs.file-max = 65536' | sudo tee -a /etc/sysctl.conf

# TCP 버퍼 최적화  
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf

# 가상 메모리 최적화
echo 'vm.max_map_count = 262144' | sudo tee -a /etc/sysctl.conf
echo 'vm.swappiness = 1' | sudo tee -a /etc/sysctl.conf

# 설정 적용
sudo sysctl -p
```

## 🧪 4단계: 설치 확인 및 테스트

### 기본 연결 테스트
```bash
# Qdrant 상태 확인
curl http://localhost:6333/

# 클러스터 정보
curl http://localhost:6333/cluster

# 컬렉션 목록
curl http://localhost:6333/collections
```

### Python 클라이언트 설치 및 테스트
```bash
# Qdrant 클라이언트 설치
pip install qdrant-client

# 테스트 스크립트 실행
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
print('✅ Qdrant 연결 성공')
print(f'클러스터 정보: {client.get_cluster_info()}')
"
```

### 간단한 벡터 저장/검색 테스트
```python
# test_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

def test_qdrant_basic():
    """Qdrant 기본 기능 테스트"""
    client = QdrantClient(host="localhost", port=6333)
    
    collection_name = "test_collection"
    
    # 1. 컬렉션 생성
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )
        print("✅ 컬렉션 생성 성공")
    except Exception as e:
        print(f"컬렉션 이미 존재: {e}")
    
    # 2. 벡터 데이터 삽입
    points = [
        PointStruct(
            id=1,
            vector=np.random.rand(128).tolist(),
            payload={"text": "테스트 문서 1", "category": "tech"}
        ),
        PointStruct(
            id=2, 
            vector=np.random.rand(128).tolist(),
            payload={"text": "테스트 문서 2", "category": "science"}
        )
    ]
    
    client.upsert(collection_name=collection_name, points=points)
    print("✅ 벡터 삽입 성공")
    
    # 3. 유사도 검색
    query_vector = np.random.rand(128).tolist()
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=2
    )
    
    print(f"✅ 검색 성공: {len(results)}개 결과")
    for result in results:
        print(f"   ID: {result.id}, Score: {result.score:.3f}")
    
    # 4. 컬렉션 삭제 (정리)
    client.delete_collection(collection_name)
    print("✅ 테스트 완료")

if __name__ == "__main__":
    test_qdrant_basic()
```

## 📊 5단계: 성능 모니터링

### Qdrant 내장 메트릭스
```bash
# 컬렉션 정보 조회
curl http://localhost:6333/collections/{collection_name}

# 클러스터 상태
curl http://localhost:6333/cluster

# 메트릭스 (Prometheus 형식)
curl http://localhost:6333/metrics
```

### 시스템 리소스 모니터링
```bash
#!/bin/bash
# monitor_qdrant.sh

echo "=== Qdrant 성능 모니터링 ==="
echo "시간: $(date)"

# Qdrant 프로세스 확인
echo ""
echo "🔧 Qdrant 프로세스:"
ps aux | grep qdrant | grep -v grep

# 메모리 사용량
echo ""
echo "💾 메모리 사용량:"
free -h

# 디스크 사용량
echo ""
echo "💿 Qdrant 데이터 크기:"
du -sh ~/qdrant_data/ 2>/dev/null || du -sh /var/lib/qdrant/ 2>/dev/null

# 네트워크 연결
echo ""
echo "🌐 네트워크 연결:"
netstat -tlnp | grep :6333

# Qdrant API 응답 시간
echo ""
echo "⚡ API 응답 시간:"
time curl -s http://localhost:6333/ > /dev/null

echo "================================="
```

### 성능 벤치마크 스크립트
```python
# benchmark_qdrant.py
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def benchmark_qdrant(num_vectors=1000, vector_size=1024):
    """Qdrant 성능 벤치마크"""
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "benchmark_collection"
    
    # 컬렉션 생성
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    
    print(f"🚀 벤치마크 시작: {num_vectors}개 벡터, {vector_size}차원")
    
    # 1. 삽입 성능 테스트
    print("\n📝 삽입 성능 테스트...")
    points = []
    for i in range(num_vectors):
        points.append(PointStruct(
            id=i,
            vector=np.random.rand(vector_size).astype(np.float32).tolist(),
            payload={"index": i, "text": f"Document {i}"}
        ))
    
    start_time = time.time()
    
    # 배치 삽입 (N100 최적화)
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        
        if i % 500 == 0:
            print(f"   진행률: {i}/{num_vectors}")
    
    insert_time = time.time() - start_time
    insert_rate = num_vectors / insert_time
    
    print(f"✅ 삽입 완료: {insert_time:.2f}초")
    print(f"📊 삽입 속도: {insert_rate:.1f} 벡터/초")
    
    # 2. 검색 성능 테스트
    print("\n🔍 검색 성능 테스트...")
    search_times = []
    
    for i in range(10):
        query_vector = np.random.rand(vector_size).astype(np.float32).tolist()
        
        start_time = time.time()
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10
        )
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        print(f"   검색 {i+1}: {search_time*1000:.1f}ms ({len(results)}개 결과)")
    
    avg_search_time = sum(search_times) / len(search_times)
    print(f"📊 평균 검색 시간: {avg_search_time*1000:.1f}ms")
    
    # 3. 메모리 사용량 확인
    import psutil
    memory = psutil.virtual_memory()
    print(f"\n💾 시스템 메모리 사용률: {memory.percent:.1f}%")
    
    # 정리
    client.delete_collection(collection_name)
    print("\n✅ 벤치마크 완료")

if __name__ == "__main__":
    # N100에 맞는 적절한 크기로 테스트
    benchmark_qdrant(num_vectors=500, vector_size=1024)
```

## 🛠️ 6단계: 운영 환경 설정

### 자동 백업 스크립트
```bash
#!/bin/bash
# backup_qdrant.sh

BACKUP_DIR="/backup/qdrant"
QDRANT_DATA="/var/lib/qdrant"  # 또는 ~/qdrant_data
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Qdrant 컨테이너 일시 정지 (데이터 일관성)
docker pause qdrant_n100 2>/dev/null

# 데이터 백업
tar -czf $BACKUP_DIR/qdrant_backup_$DATE.tar.gz $QDRANT_DATA

# 컨테이너 재시작
docker unpause qdrant_n100 2>/dev/null

# 오래된 백업 삭제 (7일 이상)
find $BACKUP_DIR -name "qdrant_backup_*.tar.gz" -mtime +7 -delete

echo "✅ Qdrant 백업 완료: qdrant_backup_$DATE.tar.gz"
```

### cron으로 자동 백업 설정
```bash
# 매일 새벽 2시 백업
echo "0 2 * * * /path/to/backup_qdrant.sh" | crontab -

# 매주 일요일 새벽 3시 백업
echo "0 3 * * 0 /path/to/backup_qdrant.sh" | crontab -
```

### 로그 로테이션 설정
```bash
# /etc/logrotate.d/qdrant
/var/log/qdrant/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    postrotate
        docker kill -s USR1 qdrant_n100 2>/dev/null || true
    endscript
}
```

## 🚨 문제 해결 가이드

### 일반적인 문제들

#### 1. 포트 충돌
```bash
# 포트 사용 확인
netstat -tlnp | grep 6333

# 다른 포트로 실행
docker run -p 6334:6333 qdrant/qdrant
```

#### 2. 메모리 부족
```bash
# 스왑 설정 확인
swapon --show

# Docker 메모리 제한 확인
docker stats qdrant_n100

# 컨테이너 재시작
docker restart qdrant_n100
```

#### 3. 디스크 공간 부족
```bash
# 디스크 사용량 확인
df -h
du -sh ~/qdrant_data/*

# 오래된 백업 정리
find ~/qdrant_data -name "*.backup" -mtime +7 -delete

# 세그먼트 최적화 강제 실행
curl -X POST http://localhost:6333/collections/{collection_name}/index
```

#### 4. 성능 저하
```bash
# CPU 사용률 확인
top -p $(pgrep qdrant)

# 메모리 매핑 확인
cat /proc/$(pgrep qdrant)/status | grep VmRSS

# 설정 최적화 재적용
docker restart qdrant_n100
```

## 📈 N100 환경 최적화 팁

### 1. 메모리 최적화
- **벡터 차원 축소**: 1024 → 512차원으로 줄이면 메모리 50% 절약
- **배치 크기 조정**: 삽입/검색 시 배치 크기를 100-200으로 제한
- **캐시 조정**: HNSW ef_construct를 100으로 낮춤

### 2. CPU 최적화  
- **스레드 제한**: max_search_threads=2로 설정
- **인덱스 최적화**: 백그라운드 최적화를 야간에만 실행
- **동시성 제어**: 동시 요청을 2-3개로 제한

### 3. 스토리지 최적화
- **SSD 사용**: HDD 대비 10배 이상 성능 향상
- **세그먼트 크기**: 200MB로 제한하여 메모리 효율성 확보
- **압축 활성화**: 백업 시 압축으로 공간 절약

### 4. 네트워크 최적화
- **로컬 호스트**: 가능한 같은 머신에서 실행
- **연결 풀링**: 클라이언트에서 연결 재사용
- **배치 요청**: 개별 요청 대신 배치 처리

## 🎯 다음 단계

1. **프로덕션 배포**: Docker Swarm 또는 Kubernetes 고려
2. **모니터링**: Prometheus + Grafana로 메트릭 수집
3. **백업 전략**: 정기 백업 및 복구 테스트 
4. **보안 설정**: API 키 및 네트워크 보안 구성
5. **스케일링**: 데이터 증가 시 클러스터 구성 고려

---

✅ **Qdrant 설치 완료 체크리스트**
- [ ] Docker 또는 네이티브 Qdrant 설치
- [ ] N100 최적화 설정 적용
- [ ] 기본 연결 테스트 성공
- [ ] Python 클라이언트 설치 및 테스트
- [ ] 성능 벤치마크 실행
- [ ] 백업 스크립트 설정
- [ ] 모니터링 스크립트 설정

🎉 **축하합니다! Qdrant 벡터 데이터베이스가 준비되었습니다.**
