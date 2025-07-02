#!/usr/bin/env python3
"""
N100 CPU 최적화된 DocLing + Ollama bge-m3 RAG 시스템
- 실제 프로덕션 환경에서 사용 가능한 완전한 구현
- 메모리 효율성과 성능 최적화에 중점
"""

import asyncio
import time
import logging
import psutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

# DocLing 임포트
try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    DOCLING_AVAILABLE = True
except ImportError:
    print("⚠️ DocLing을 설치하세요: pip install docling")
    DOCLING_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    # Ollama 설정
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    
    # bge-m3 토크나이저 설정
    bge_model_id: str = "BAAI/bge-m3"
    max_tokens: int = 512  # N100 최적화
    
    # 청킹 설정
    chunk_overlap: int = 50
    merge_peers: bool = True
    merge_list_items: bool = True
    
    # 성능 설정
    batch_size: int = 3
    max_parallel: int = 2
    timeout: int = 15
    cache_size: int = 1000
    
    # 검색 설정
    top_k: int = 5
    similarity_threshold: float = 0.3

@dataclass
class Document:
    """문서 클래스"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None

@dataclass
class SearchResult:
    """검색 결과 클래스"""
    text: str
    similarity: float
    metadata: Dict[str, Any]
    chunk_id: str

class OllamaEmbeddingClient:
    """Ollama 임베딩 클라이언트"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.session = requests.Session()
        
        # 연결 풀 최적화
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=config.max_parallel,
            pool_maxsize=config.max_parallel * 2,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # 캐시
        self._embedding_cache = {}
        
    @lru_cache(maxsize=1000)
    def _cache_key(self, text: str) -> str:
        """캐시 키 생성"""
        return f"bge_m3_{hash(text[:100])}"  # 첫 100자만 해시
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """단일 텍스트 임베딩"""
        cache_key = self._cache_key(text)
        
        # 캐시 확인
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            # 토큰 수 제한
            limited_text = text[:self.config.max_tokens * 4]  # 대략적 계산
            
            response = self.session.post(
                f"{self.config.ollama_base_url}/api/embeddings",
                json={
                    "model": self.config.embed_model,
                    "prompt": limited_text
                },
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = np.array(data["embedding"], dtype=np.float32)
                
                # 캐시 저장 (크기 제한)
                if len(self._embedding_cache) < self.config.cache_size:
                    self._embedding_cache[cache_key] = embedding
                
                return embedding
            else:
                logger.error(f"Ollama 임베딩 실패: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"임베딩 요청 오류: {e}")
            return None
    
    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """배치 임베딩"""
        with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
            futures = [executor.submit(self.embed_text, text) for text in texts]
            results = []
            
            for future in futures:
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                except Exception as e:
                    logger.error(f"배치 임베딩 오류: {e}")
                    results.append(None)
            
            return results
    
    def check_ollama_status(self) -> bool:
        """Ollama 서버 상태 확인"""
        try:
            response = self.session.get(
                f"{self.config.ollama_base_url}/api/tags", 
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.config.embed_model in model_names:
                    logger.info(f"✅ {self.config.embed_model} 모델 사용 가능")
                    return True
                else:
                    logger.warning(f"❌ {self.config.embed_model} 모델 없음")
                    logger.info(f"사용 가능한 모델: {model_names}")
                    logger.info(f"설치 명령: ollama pull {self.config.embed_model}")
                    return False
            else:
                logger.error(f"Ollama API 응답 오류: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Ollama 연결 실패: {e}")
            logger.info("Ollama 서버가 실행 중인지 확인하세요: ollama serve")
            return False

class DoclingProcessor:
    """DocLing 문서 처리기"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        if not DOCLING_AVAILABLE:
            raise ImportError("DocLing이 설치되지 않았습니다.")
        
        # 토크나이저 초기화
        try:
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(config.bge_model_id),
                max_tokens=config.max_tokens
            )
            
            # HybridChunker 초기화
            self.chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=config.merge_peers,
                merge_list_items=config.merge_list_items
            )
            
            # 문서 변환기
            self.converter = DocumentConverter()
            
            logger.info("✅ DocLing 프로세서 초기화 완료")
            
        except Exception as e:
            logger.error(f"DocLing 초기화 실패: {e}")
            raise
    
    def process_file(self, file_path: str) -> List[str]:
        """파일을 처리하고 컨텍스트화된 청크 반환"""
        try:
            # 문서 변환
            logger.info(f"📄 파일 처리 중: {file_path}")
            converted_doc = self.converter.convert(source=file_path)
            
            # 청킹 및 컨텍스트화
            chunks = []
            for chunk in self.chunker.chunk(dl_doc=converted_doc.document):
                # contextualize 적용 (실제로는 serialize)
                contextualized_text = self.chunker.serialize(chunk)
                chunks.append(contextualized_text)
            
            logger.info(f"✅ {len(chunks)}개 청크 생성")
            return chunks
            
        except Exception as e:
            logger.error(f"파일 처리 실패 {file_path}: {e}")
            return []
    
    def process_text(self, text: str) -> List[str]:
        """일반 텍스트를 처리하고 청크 반환"""
        try:
            # 임시 문서 객체 생성 (실제 구현에서는 더 정교해야 함)
            # 여기서는 간단한 문단 분할 사용
            paragraphs = text.split('\n\n')
            chunks = []
            
            for para in paragraphs:
                if len(para.strip()) > 0:
                    # 긴 문단은 문장 단위로 분할
                    if len(para) > self.config.max_tokens * 3:
                        sentences = para.split('. ')
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) < self.config.max_tokens * 3:
                                current_chunk += sentence + ". "
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sentence + ". "
                        
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                    else:
                        chunks.append(para.strip())
            
            return chunks
            
        except Exception as e:
            logger.error(f"텍스트 처리 실패: {e}")
            return []

class N100OptimizedRAG:
    """N100 최적화된 완전한 RAG 시스템"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # 구성 요소 초기화
        self.embedding_client = OllamaEmbeddingClient(config)
        
        if DOCLING_AVAILABLE:
            self.docling_processor = DoclingProcessor(config)
        else:
            self.docling_processor = None
            logger.warning("DocLing 비활성화 - 텍스트 전용 모드")
        
        # 문서 저장소
        self.documents: Dict[str, Document] = {}
        self.chunk_embeddings: Dict[str, Dict[str, Any]] = {}
        
        # 성능 모니터링
        self.performance_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'searches_performed': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def add_document_from_file(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """파일에서 문서 추가"""
        if not self.docling_processor:
            logger.error("DocLing 프로세서가 없습니다.")
            return False
        
        try:
            # 파일 처리
            chunks = self.docling_processor.process_file(file_path)
            
            if not chunks:
                logger.warning(f"파일에서 청크를 생성할 수 없습니다: {file_path}")
                return False
            
            # 문서 추가
            doc = Document(
                content="\n\n".join(chunks),
                metadata=metadata or {},
                file_path=file_path
            )
            
            return self._add_document_chunks(file_path, chunks, doc.metadata)
            
        except Exception as e:
            logger.error(f"파일 문서 추가 실패: {e}")
            return False
    
    def add_document_from_text(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> bool:
        """텍스트에서 문서 추가"""
        try:
            # 텍스트 처리
            if self.docling_processor:
                chunks = self.docling_processor.process_text(text)
            else:
                # 간단한 청킹
                chunks = self._simple_chunk_text(text)
            
            if not chunks:
                logger.warning(f"텍스트에서 청크를 생성할 수 없습니다: {doc_id}")
                return False
            
            # 문서 추가
            doc = Document(
                content=text,
                metadata=metadata or {}
            )
            
            return self._add_document_chunks(doc_id, chunks, doc.metadata)
            
        except Exception as e:
            logger.error(f"텍스트 문서 추가 실패: {e}")
            return False
    
    def _simple_chunk_text(self, text: str) -> List[str]:
        """간단한 텍스트 청킹 (DocLing 없을 때)"""
        max_chars = self.config.max_tokens * 3  # 대략적 계산
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _add_document_chunks(self, doc_id: str, chunks: List[str], metadata: Dict) -> bool:
        """청크들을 임베딩하고 저장"""
        try:
            logger.info(f"📝 {len(chunks)}개 청크 임베딩 중...")
            
            # 배치 단위로 임베딩
            all_embeddings = []
            
            for i in range(0, len(chunks), self.config.batch_size):
                batch = chunks[i:i + self.config.batch_size]
                batch_embeddings = self.embedding_client.embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # N100 열 관리
                time.sleep(0.3)
                
                progress = min(i + self.config.batch_size, len(chunks))
                logger.info(f"   진행률: {progress}/{len(chunks)}")
            
            # 결과 저장
            successful_chunks = 0
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                if embedding is not None:
                    chunk_id = f"{doc_id}_chunk_{i}"
                    
                    self.chunk_embeddings[chunk_id] = {
                        'embedding': embedding,
                        'text': chunk,
                        'metadata': {
                            **metadata,
                            'doc_id': doc_id,
                            'chunk_index': i,
                            'chunk_id': chunk_id
                        }
                    }
                    successful_chunks += 1
            
            # 통계 업데이트
            self.performance_stats['documents_processed'] += 1
            self.performance_stats['chunks_created'] += successful_chunks
            
            logger.info(f"✅ {successful_chunks}/{len(chunks)} 청크 성공적으로 추가")
            return successful_chunks > 0
            
        except Exception as e:
            logger.error(f"청크 임베딩 실패: {e}")
            return False
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """문서 검색"""
        start_time = time.time()
        top_k = top_k or self.config.top_k
        
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_client.embed_text(query)
            
            if query_embedding is None:
                logger.error("쿼리 임베딩 실패")
                return []
            
            # 유사도 계산
            similarities = []
            for chunk_id, chunk_data in self.chunk_embeddings.items():
                similarity = self._cosine_similarity(
                    query_embedding, 
                    chunk_data['embedding']
                )
                
                # 임계값 필터링
                if similarity >= self.config.similarity_threshold:
                    similarities.append({
                        'chunk_id': chunk_id,
                        'similarity': similarity,
                        'text': chunk_data['text'],
                        'metadata': chunk_data['metadata']
                    })
            
            # 상위 k개 정렬
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:top_k]
            
            # SearchResult 객체로 변환
            results = [
                SearchResult(
                    text=result['text'],
                    similarity=result['similarity'],
                    metadata=result['metadata'],
                    chunk_id=result['chunk_id']
                ) for result in top_results
            ]
            
            # 성능 통계 업데이트
            search_time = time.time() - start_time
            self.performance_stats['searches_performed'] += 1
            
            current_avg = self.performance_stats['avg_search_time']
            searches_count = self.performance_stats['searches_performed']
            self.performance_stats['avg_search_time'] = (
                (current_avg * (searches_count - 1) + search_time) / searches_count
            )
            
            logger.info(f"🔍 검색 완료: {len(results)}개 결과, {search_time:.3f}초")
            return results
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'documents_count': self.performance_stats['documents_processed'],
            'chunks_count': self.performance_stats['chunks_created'],
            'searches_performed': self.performance_stats['searches_performed'],
            'avg_search_time': self.performance_stats['avg_search_time'],
            'cache_size': len(self.embedding_client._embedding_cache),
            'memory_usage_percent': memory.percent,
            'cpu_usage_percent': cpu_percent,
            'memory_available_gb': memory.available / (1024**3),
            'ollama_connected': self.embedding_client.check_ollama_status()
        }
    
    def optimize_for_n100(self):
        """N100 환경 최적화 실행"""
        logger.info("🔧 N100 환경 최적화 적용 중...")
        
        # 가비지 컬렉션
        import gc
        gc.collect()
        
        # 캐시 크기 조정
        cache_size = len(self.embedding_client._embedding_cache)
        if cache_size > self.config.cache_size * 0.8:
            # 캐시 정리 (LRU 방식)
            cache_items = list(self.embedding_client._embedding_cache.items())
            keep_size = self.config.cache_size // 2
            self.embedding_client._embedding_cache = dict(cache_items[-keep_size:])
            logger.info(f"🧹 캐시 정리: {cache_size} → {keep_size}")
        
        # 메모리 사용량 체크
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            logger.warning(f"⚠️ 높은 메모리 사용량: {memory.percent:.1f}%")
        
        logger.info("✅ N100 최적화 완료")

# 실제 사용 예제와 테스트
class RAGSystemTester:
    """RAG 시스템 테스터"""
    
    def __init__(self, rag_system: N100OptimizedRAG):
        self.rag = rag_system
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        logger.info("🧪 RAG 시스템 종합 테스트 시작")
        
        # 1. Ollama 연결 테스트
        self._test_ollama_connection()
        
        # 2. 문서 추가 테스트
        self._test_document_addition()
        
        # 3. 검색 테스트
        self._test_search_functionality()
        
        # 4. 성능 테스트
        self._test_performance()
        
        # 5. 시스템 상태 확인
        self._check_system_status()
    
    def _test_ollama_connection(self):
        """Ollama 연결 테스트"""
        logger.info("🔌 Ollama 연결 테스트...")
        
        if self.rag.embedding_client.check_ollama_status():
            logger.info("✅ Ollama 연결 성공")
        else:
            logger.error("❌ Ollama 연결 실패")
            return False
        
        # 간단한 임베딩 테스트
        test_embedding = self.rag.embedding_client.embed_text("테스트")
        if test_embedding is not None:
            logger.info(f"✅ 임베딩 테스트 성공 (차원: {len(test_embedding)})")
        else:
            logger.error("❌ 임베딩 테스트 실패")
        
        return True
    
    def _test_document_addition(self):
        """문서 추가 테스트"""
        logger.info("📚 문서 추가 테스트...")
        
        # 샘플 문서들
        test_documents = [
            {
                'id': 'doc1',
                'text': "IBM은 1911년에 설립된 미국의 다국적 기술 회사입니다. 인공지능, 클라우드 컴퓨팅, 양자 컴퓨팅 분야에서 선도적인 역할을 하고 있습니다. 1960년대부터 메인프레임 컴퓨터 시장을 주도했으며, System/360으로 컴퓨터 산업을 혁신했습니다.",
                'metadata': {'source': 'wiki', 'topic': 'IBM', 'language': 'ko'}
            },
            {
                'id': 'doc2', 
                'text': "DocLing은 IBM Research에서 개발한 오픈소스 문서 변환 도구입니다. PDF, DOCX, PPTX 등 다양한 형식의 문서를 구조화된 데이터로 변환할 수 있습니다. HybridChunker를 통해 문맥을 보존하는 청킹 기능을 제공하며, contextualize 메소드로 각 청크에 문서의 계층적 구조 정보를 추가합니다.",
                'metadata': {'source': 'docs', 'topic': 'DocLing', 'language': 'ko'}
            },
            {
                'id': 'doc3',
                'text': "bge-m3는 BAAI(Beijing Academy of Artificial Intelligence)에서 개발한 다국어 임베딩 모델입니다. 한국어, 영어, 중국어 등 100여 개 언어를 지원하며, 최대 8192 토큰까지 처리할 수 있습니다. MTEB(Massive Text Embedding Benchmark)에서 우수한 성능을 보여주며, 특히 다국어 검색 작업에서 뛰어난 결과를 보입니다.",
                'metadata': {'source': 'research', 'topic': 'bge-m3', 'language': 'ko'}
            }
        ]
        
        success_count = 0
        for doc in test_documents:
            if self.rag.add_document_from_text(doc['id'], doc['text'], doc['metadata']):
                success_count += 1
                logger.info(f"✅ 문서 추가 성공: {doc['id']}")
            else:
                logger.error(f"❌ 문서 추가 실패: {doc['id']}")
        
        logger.info(f"📊 문서 추가 결과: {success_count}/{len(test_documents)}")
        return success_count == len(test_documents)
    
    def _test_search_functionality(self):
        """검색 기능 테스트"""
        logger.info("🔍 검색 기능 테스트...")
        
        test_queries = [
            "IBM의 역사와 System/360에 대해 알려주세요",
            "DocLing의 HybridChunker와 contextualize 기능은 무엇인가요?",
            "bge-m3 모델의 특징과 지원 언어는?",
            "양자 컴퓨팅 관련 정보를 찾아주세요",
            "다국어 임베딩 모델의 성능은 어떤가요?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 쿼리 {i}: {query}")
            
            results = self.rag.search(query, top_k=3)
            
            if results:
                logger.info(f"📄 {len(results)}개 결과 발견:")
                for j, result in enumerate(results, 1):
                    logger.info(f"  {j}. 유사도: {result.similarity:.3f}")
                    logger.info(f"     내용: {result.text[:100]}...")
                    logger.info(f"     주제: {result.metadata.get('topic', 'N/A')}")
            else:
                logger.warning("❌ 검색 결과 없음")
    
    def _test_performance(self):
        """성능 테스트"""
        logger.info("⚡ 성능 테스트...")
        
        # 반복 검색으로 캐시 효과 측정
        test_query = "IBM 기술"
        times = []
        
        for i in range(5):
            start_time = time.time()
            results = self.rag.search(test_query)
            search_time = time.time() - start_time
            times.append(search_time)
            logger.info(f"  검색 {i+1}: {search_time:.3f}초")
        
        avg_time = sum(times) / len(times)
        logger.info(f"📊 평균 검색 시간: {avg_time:.3f}초")
        
        # 메모리 사용량
        memory = psutil.virtual_memory()
        logger.info(f"💾 메모리 사용률: {memory.percent:.1f}%")
        
        # N100 최적화 실행
        self.rag.optimize_for_n100()
    
    def _check_system_status(self):
        """시스템 상태 확인"""
        logger.info("📊 시스템 상태 확인...")
        
        status = self.rag.get_system_status()
        
        logger.info(f"📈 시스템 통계:")
        logger.info(f"  - 처리된 문서: {status['documents_count']}")
        logger.info(f"  - 생성된 청크: {status['chunks_count']}")
        logger.info(f"  - 수행된 검색: {status['searches_performed']}")
        logger.info(f"  - 평균 검색 시간: {status['avg_search_time']:.3f}초")
        logger.info(f"  - 캐시 크기: {status['cache_size']}")
        logger.info(f"  - 메모리 사용률: {status['memory_usage_percent']:.1f}%")
        logger.info(f"  - CPU 사용률: {status['cpu_usage_percent']:.1f}%")
        logger.info(f"  - 사용 가능 메모리: {status['memory_available_gb']:.1f}GB")

def setup_and_run_demo():
    """데모 설정 및 실행"""
    print("🚀 N100 + DocLing + Ollama bge-m3 RAG 시스템 데모")
    print("=" * 60)
    
    # 설정
    config = RAGConfig(
        max_tokens=512,      # N100용 최적화
        batch_size=3,        # 작은 배치
        max_parallel=2,      # 쿼드코어의 절반
        cache_size=500,      # 메모리 절약
        top_k=5
    )
    
    # RAG 시스템 초기화
    try:
        rag_system = N100OptimizedRAG(config)
        logger.info("✅ RAG 시스템 초기화 완료")
    except Exception as e:
        logger.error(f"❌ RAG 시스템 초기화 실패: {e}")
        return
    
    # 테스터 실행
    tester = RAGSystemTester(rag_system)
    tester.run_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("🎯 N100 환경 최적화 팁:")
    print("1. Ollama 설정: export OLLAMA_NUM_PARALLEL=2")
    print("2. 모델 설치: ollama pull bge-m3")
    print("3. 메모리 모니터링: watch -n 1 'free -h'")
    print("4. 온도 확인: sensors (lm-sensors 패키지)")
    print("5. 스왑 설정: sudo swapon /swapfile (필요시)")

if __name__ == "__main__":
    # 로그 레벨 설정
    logging.getLogger().setLevel(logging.INFO)
    
    # 데모 실행
    setup_and_run_demo()
