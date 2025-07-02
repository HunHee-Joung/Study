#!/usr/bin/env python3
"""
N100 최적화된 DocLing + Ollama bge-m3 + Qdrant RAG 시스템
- 파일명을 메타데이터로 활용
- Qdrant 벡터 데이터베이스 통합
- 고성능 검색 및 필터링 지원
"""

import asyncio
import time
import logging
import psutil
import uuid
import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

# Qdrant 임포트
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, 
        FieldCondition, Match, MatchValue, SearchParams
    )
    QDRANT_AVAILABLE = True
except ImportError:
    print("⚠️ Qdrant를 설치하세요: pip install qdrant-client")
    QDRANT_AVAILABLE = False

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
class QdrantRAGConfig:
    """Qdrant RAG 시스템 설정"""
    # Qdrant 설정
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    collection_name: str = "rag_documents"
    vector_size: int = 1024  # bge-m3 임베딩 차원
    
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
    top_k: int = 10
    similarity_threshold: float = 0.3
    
    # 파일 처리 설정
    supported_extensions: List[str] = field(default_factory=lambda: [
        '.pdf', '.docx', '.pptx', '.txt', '.md', '.html'
    ])

@dataclass
class DocumentMetadata:
    """강화된 문서 메타데이터"""
    file_name: str
    file_path: str
    file_extension: str
    file_size: int
    created_at: float
    processed_at: float
    chunk_index: int
    total_chunks: int
    doc_id: str
    chunk_id: str
    
    # 추가 메타데이터
    source_type: str = "file"  # file, text, url
    language: str = "auto"
    topic: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (Qdrant 호환)"""
        return {
            'file_name': self.file_name,
            'file_path': self.file_path,
            'file_extension': self.file_extension,
            'file_size': self.file_size,
            'created_at': self.created_at,
            'processed_at': self.processed_at,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'doc_id': self.doc_id,
            'chunk_id': self.chunk_id,
            'source_type': self.source_type,
            'language': self.language,
            'topic': self.topic,
            'author': self.author,
            'tags': self.tags
        }

@dataclass
class SearchResult:
    """검색 결과 클래스"""
    text: str
    similarity: float
    metadata: DocumentMetadata
    qdrant_id: str

class QdrantManager:
    """Qdrant 벡터 데이터베이스 관리자"""
    
    def __init__(self, config: QdrantRAGConfig):
        self.config = config
        
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client가 설치되지 않았습니다.")
        
        # Qdrant 클라이언트 초기화
        try:
            self.client = QdrantClient(
                host=config.qdrant_host,
                port=config.qdrant_port,
                api_key=config.qdrant_api_key,
                timeout=config.timeout
            )
            
            # 컬렉션 초기화
            self._initialize_collection()
            
            logger.info(f"✅ Qdrant 연결 성공: {config.qdrant_host}:{config.qdrant_port}")
            
        except Exception as e:
            logger.error(f"❌ Qdrant 연결 실패: {e}")
            raise
    
    def _initialize_collection(self):
        """컬렉션 초기화"""
        try:
            # 컬렉션 존재 확인
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.config.collection_name not in collection_names:
                # 새 컬렉션 생성
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"📁 새 컬렉션 생성: {self.config.collection_name}")
            else:
                logger.info(f"📁 기존 컬렉션 사용: {self.config.collection_name}")
                
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {e}")
            raise
    
    def add_points(self, points: List[PointStruct]) -> bool:
        """포인트 배치 추가"""
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
            logger.info(f"✅ {len(points)}개 포인트 추가 완료")
            return True
            
        except Exception as e:
            logger.error(f"포인트 추가 실패: {e}")
            return False
    
    def search_similar(self, 
                      query_vector: List[float], 
                      limit: int = 10,
                      score_threshold: Optional[float] = None,
                      filter_conditions: Optional[Filter] = None) -> List[Dict]:
        """유사도 검색"""
        try:
            search_params = SearchParams(
                hnsw_ef=128,  # N100 최적화
                exact=False
            )
            
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
                search_params=search_params,
                with_payload=True,
                with_vectors=False
            )
            
            return [
                {
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []
    
    def delete_by_file(self, file_name: str) -> bool:
        """파일명으로 모든 관련 포인트 삭제"""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="file_name",
                        match=MatchValue(value=file_name)
                    )
                ]
            )
            
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"🗑️ 파일 관련 데이터 삭제: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"파일 삭제 실패: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                'name': info.config.name,
                'vector_size': info.config.params.vectors.size,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}

class FileMetadataExtractor:
    """파일 메타데이터 추출기"""
    
    @staticmethod
    def extract_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
        """파일에서 메타데이터 추출"""
        file_path = Path(file_path)
        
        try:
            stat = file_path.stat()
            
            # 기본 메타데이터
            metadata = {
                'file_name': file_path.name,
                'file_path': str(file_path.absolute()),
                'file_extension': file_path.suffix.lower(),
                'file_size': stat.st_size,
                'created_at': stat.st_ctime,
                'processed_at': time.time()
            }
            
            # 파일명에서 추가 정보 추출
            file_stem = file_path.stem
            
            # 날짜 패턴 감지 (YYYY-MM-DD, YYYYMMDD 등)
            date_patterns = [
                r'(\d{4})-(\d{2})-(\d{2})',
                r'(\d{4})(\d{2})(\d{2})',
                r'(\d{4})\.(\d{2})\.(\d{2})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, file_stem)
                if match:
                    metadata['extracted_date'] = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                    break
            
            # 버전 정보 감지 (v1.0, version_2 등)
            version_pattern = r'v?(\d+)\.?(\d+)?\.?(\d+)?'
            version_match = re.search(version_pattern, file_stem.lower())
            if version_match:
                metadata['version'] = version_match.group(0)
            
            # 언어 감지 (파일명에서)
            language_indicators = {
                'ko': ['한국어', 'korean', '한글', 'kor'],
                'en': ['english', 'eng', 'en'],
                'zh': ['chinese', 'china', 'zh', 'cn'],
                'ja': ['japanese', 'japan', 'jp']
            }
            
            file_lower = file_stem.lower()
            for lang_code, indicators in language_indicators.items():
                if any(indicator in file_lower for indicator in indicators):
                    metadata['detected_language'] = lang_code
                    break
            
            # 문서 타입 추정
            doc_type_keywords = {
                'manual': ['manual', '매뉴얼', '설명서', 'guide'],
                'report': ['report', '보고서', 'analysis', '분석'],
                'presentation': ['presentation', '발표', 'ppt'],
                'specification': ['spec', '명세서', 'specification'],
                'tutorial': ['tutorial', '튜토리얼', 'howto', '방법']
            }
            
            for doc_type, keywords in doc_type_keywords.items():
                if any(keyword in file_lower for keyword in keywords):
                    metadata['document_type'] = doc_type
                    break
            
            return metadata
            
        except Exception as e:
            logger.error(f"메타데이터 추출 실패 {file_path}: {e}")
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_extension': file_path.suffix.lower(),
                'file_size': 0,
                'created_at': time.time(),
                'processed_at': time.time()
            }

class OllamaEmbeddingClient:
    """Ollama 임베딩 클라이언트 (캐싱 강화)"""
    
    def __init__(self, config: QdrantRAGConfig):
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
        
        # 캐시 (메모리 + 간단한 파일 캐시)
        self._memory_cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
        
    @lru_cache(maxsize=1000)
    def _cache_key(self, text: str) -> str:
        """캐시 키 생성"""
        return f"bge_m3_{hash(text[:200])}"  # 첫 200자 해시
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """단일 텍스트 임베딩 (캐싱 포함)"""
        cache_key = self._cache_key(text)
        
        # 메모리 캐시 확인
        if cache_key in self._memory_cache:
            self._cache_stats['hits'] += 1
            return self._memory_cache[cache_key]
        
        self._cache_stats['misses'] += 1
        
        try:
            # 토큰 수 제한
            limited_text = text[:self.config.max_tokens * 4]
            
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
                
                # 메모리 캐시 저장 (크기 제한)
                if len(self._memory_cache) < self.config.cache_size:
                    self._memory_cache[cache_key] = embedding
                
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = self._cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_stats['hits'],
            'cache_misses': self._cache_stats['misses'],
            'hit_rate': hit_rate,
            'cache_size': len(self._memory_cache)
        }

class DoclingProcessor:
    """DocLing 문서 처리기 (파일 메타데이터 통합)"""
    
    def __init__(self, config: QdrantRAGConfig):
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
    
    def process_file(self, file_path: Union[str, Path]) -> Tuple[List[str], Dict[str, Any]]:
        """파일 처리 및 메타데이터 추출"""
        file_path = Path(file_path)
        
        try:
            # 파일 메타데이터 추출
            file_metadata = FileMetadataExtractor.extract_file_metadata(file_path)
            
            # 확장자 확인
            if file_metadata['file_extension'] not in self.config.supported_extensions:
                logger.warning(f"지원하지 않는 파일 형식: {file_metadata['file_extension']}")
                return [], file_metadata
            
            logger.info(f"📄 파일 처리 중: {file_metadata['file_name']}")
            
            # 문서 변환
            converted_doc = self.converter.convert(source=str(file_path))
            
            # 청킹 및 컨텍스트화
            chunks = []
            for chunk in self.chunker.chunk(dl_doc=converted_doc.document):
                # contextualize 적용 (serialize 메소드 사용)
                contextualized_text = self.chunker.serialize(chunk)
                chunks.append(contextualized_text)
            
            logger.info(f"✅ {len(chunks)}개 청크 생성")
            return chunks, file_metadata
            
        except Exception as e:
            logger.error(f"파일 처리 실패 {file_path}: {e}")
            return [], FileMetadataExtractor.extract_file_metadata(file_path)

class QdrantRAGSystem:
    """Qdrant 기반 완전한 RAG 시스템"""
    
    def __init__(self, config: QdrantRAGConfig):
        self.config = config
        
        # 구성 요소 초기화
        self.qdrant_manager = QdrantManager(config)
        self.embedding_client = OllamaEmbeddingClient(config)
        
        if DOCLING_AVAILABLE:
            self.docling_processor = DoclingProcessor(config)
        else:
            self.docling_processor = None
            logger.warning("DocLing 비활성화 - 텍스트 전용 모드")
        
        # 성능 통계
        self.performance_stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'searches_performed': 0,
            'avg_search_time': 0.0,
            'total_processing_time': 0.0
        }
    
    def add_file(self, file_path: Union[str, Path], 
                 additional_metadata: Optional[Dict] = None) -> bool:
        """파일 추가 (전체 파이프라인)"""
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            if not file_path.exists():
                logger.error(f"파일이 존재하지 않습니다: {file_path}")
                return False
            
            # DocLing으로 파일 처리
            if self.docling_processor:
                chunks, file_metadata = self.docling_processor.process_file(file_path)
            else:
                # 텍스트 파일 직접 처리
                chunks, file_metadata = self._process_text_file(file_path)
            
            if not chunks:
                logger.warning(f"처리할 청크가 없습니다: {file_path}")
                return False
            
            # 추가 메타데이터 병합
            if additional_metadata:
                file_metadata.update(additional_metadata)
            
            # 문서 ID 생성
            doc_id = str(uuid.uuid4())
            
            # 청크 임베딩 및 Qdrant 저장
            success = self._embed_and_store_chunks(chunks, file_metadata, doc_id)
            
            if success:
                # 통계 업데이트
                processing_time = time.time() - start_time
                self.performance_stats['files_processed'] += 1
                self.performance_stats['chunks_created'] += len(chunks)
                self.performance_stats['total_processing_time'] += processing_time
                
                logger.info(f"✅ 파일 처리 완료: {file_metadata['file_name']} ({processing_time:.2f}초)")
                return True
            else:
                logger.error(f"❌ 파일 처리 실패: {file_metadata['file_name']}")
                return False
                
        except Exception as e:
            logger.error(f"파일 추가 중 오류: {e}")
            return False
    
    def _process_text_file(self, file_path: Path) -> Tuple[List[str], Dict[str, Any]]:
        """텍스트 파일 직접 처리"""
        try:
            # 파일 메타데이터
            file_metadata = FileMetadataExtractor.extract_file_metadata(file_path)
            
            # 텍스트 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 간단한 청킹
            chunks = self._simple_chunk_text(content)
            
            return chunks, file_metadata
            
        except Exception as e:
            logger.error(f"텍스트 파일 처리 실패: {e}")
            return [], FileMetadataExtractor.extract_file_metadata(file_path)
    
    def _simple_chunk_text(self, text: str) -> List[str]:
        """간단한 텍스트 청킹"""
        max_chars = self.config.max_tokens * 3
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chars:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _embed_and_store_chunks(self, chunks: List[str], 
                               file_metadata: Dict[str, Any], 
                               doc_id: str) -> bool:
        """청크 임베딩 및 Qdrant 저장"""
        try:
            logger.info(f"📝 {len(chunks)}개 청크 임베딩 중...")
            
            # 배치 단위로 임베딩
            all_embeddings = []
            for i in range(0, len(chunks), self.config.batch_size):
                batch = chunks[i:i + self.config.batch_size]
                batch_embeddings = self.embedding_client.embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # N100 열 관리
                time.sleep(0.2)
                
                progress = min(i + self.config.batch_size, len(chunks))
                logger.info(f"   임베딩 진행률: {progress}/{len(chunks)}")
            
            # Qdrant 포인트 생성
            points = []
            successful_chunks = 0
            
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                if embedding is not None:
                    chunk_id = f"{doc_id}_chunk_{i}"
                    
                    # 메타데이터 객체 생성
                    metadata = DocumentMetadata(
                        file_name=file_metadata['file_name'],
                        file_path=file_metadata['file_path'],
                        file_extension=file_metadata['file_extension'],
                        file_size=file_metadata['file_size'],
                        created_at=file_metadata['created_at'],
                        processed_at=file_metadata['processed_at'],
                        chunk_index=i,
                        total_chunks=len(chunks),
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        
                        # 추가 정보
                        language=file_metadata.get('detected_language', 'auto'),
                        topic=file_metadata.get('document_type'),
                        tags=file_metadata.get('tags', [])
                    )
                    
                    # Qdrant 포인트 생성
                    point = PointStruct(
                        id=chunk_id,
                        vector=embedding.tolist(),
                        payload={
                            **metadata.to_dict(),
                            'text': chunk,
                            'text_length': len(chunk)
                        }
                    )
                    
                    points.append(point)
                    successful_chunks += 1
            
            # Qdrant에 배치 저장
            if points:
                success = self.qdrant_manager.add_points(points)
                if success:
                    logger.info(f"✅ {successful_chunks}/{len(chunks)} 청크 Qdrant 저장 완료")
                    return True
                else:
                    logger.error("Qdrant 저장 실패")
                    return False
            else:
                logger.warning("저장할 유효한 청크가 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"임베딩 및 저장 실패: {e}")
            return False
    
    def search(self, query: str, 
               top_k: Optional[int] = None,
               file_filter: Optional[str] = None,
               extension_filter: Optional[str] = None,
               date_range: Optional[Tuple[float, float]] = None) -> List[SearchResult]:
        """고급 검색 (필터링 지원)"""
        start_time = time.time()
        top_k = top_k or self.config.top_k
        
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_client.embed_text(query)
            if query_embedding is None:
                logger.error("쿼리 임베딩 실패")
                return []
            
            # 필터 조건 구성
            filter_conditions = []
            
            if file_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="file_name",
                        match=MatchValue(value=file_filter)
                    )
                )
            
            if extension_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="file_extension",
                        match=MatchValue(value=extension_filter)
                    )
                )
            
            if date_range:
                start_date, end_date = date_range
                filter_conditions.append(
                    FieldCondition(
                        key="created_at",
                        range={
                            "gte": start_date,
                            "lte": end_date
                        }
                    )
                )
            
            # 필터 객체 생성
            query_filter = None
            if filter_conditions:
                query_filter = Filter(must=filter_conditions)
            
            # Qdrant 검색
            search_results = self.qdrant_manager.search_similar(
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=self.config.similarity_threshold,
                filter_conditions=query_filter
            )
            
            # SearchResult 객체로 변환
            results = []
            for result in search_results:
                payload = result['payload']
                
                metadata = DocumentMetadata(
                    file_name=payload['file_name'],
                    file_path=payload['file_path'],
                    file_extension=payload['file_extension'],
                    file_size=payload['file_size'],
                    created_at=payload['created_at'],
                    processed_at=payload['processed_at'],
                    chunk_index=payload['chunk_index'],
                    total_chunks=payload['total_chunks'],
                    doc_id=payload['doc_id'],
                    chunk_id=payload['chunk_id'],
                    source_type=payload.get('source_type', 'file'),
                    language=payload.get('language', 'auto'),
                    topic=payload.get('topic'),
                    author=payload.get('author'),
                    tags=payload.get('tags', [])
                )
                
                search_result = SearchResult(
                    text=payload['text'],
                    similarity=result['score'],
                    metadata=metadata,
                    qdrant_id=result['id']
                )
                
                results.append(search_result)
            
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
    
    def search_by_filename(self, filename: str, top_k: int = 5) -> List[SearchResult]:
        """파일명으로 검색"""
        return self.search("", top_k=top_k, file_filter=filename)
    
    def search_by_extension(self, extension: str, query: str = "", top_k: int = 10) -> List[SearchResult]:
        """파일 확장자로 검색"""
        if not extension.startswith('.'):
            extension = '.' + extension
        return self.search(query, top_k=top_k, extension_filter=extension)
    
    def get_file_list(self) -> List[Dict[str, Any]]:
        """저장된 파일 목록 조회"""
        try:
            # 모든 고유 파일명 조회 (Qdrant scroll 사용)
            scroll_result = self.qdrant_manager.client.scroll(
                collection_name=self.config.collection_name,
                limit=1000,  # 적절한 크기로 조정
                with_payload=True,
                with_vectors=False
            )
            
            files = {}
            for point in scroll_result[0]:
                payload = point.payload
                file_name = payload['file_name']
                
                if file_name not in files:
                    files[file_name] = {
                        'file_name': file_name,
                        'file_path': payload['file_path'],
                        'file_extension': payload['file_extension'],
                        'file_size': payload['file_size'],
                        'created_at': payload['created_at'],
                        'processed_at': payload['processed_at'],
                        'total_chunks': payload['total_chunks'],
                        'language': payload.get('language', 'auto'),
                        'topic': payload.get('topic'),
                        'tags': payload.get('tags', [])
                    }
            
            return list(files.values())
            
        except Exception as e:
            logger.error(f"파일 목록 조회 실패: {e}")
            return []
    
    def delete_file(self, filename: str) -> bool:
        """파일 및 관련 청크 삭제"""
        try:
            success = self.qdrant_manager.delete_by_file(filename)
            if success:
                logger.info(f"🗑️ 파일 삭제 완료: {filename}")
            return success
        except Exception as e:
            logger.error(f"파일 삭제 실패: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Qdrant 정보
        qdrant_info = self.qdrant_manager.get_collection_info()
        
        # 캐시 통계
        cache_stats = self.embedding_client.get_cache_stats()
        
        return {
            'performance': {
                'files_processed': self.performance_stats['files_processed'],
                'chunks_created': self.performance_stats['chunks_created'],
                'searches_performed': self.performance_stats['searches_performed'],
                'avg_search_time': self.performance_stats['avg_search_time'],
                'total_processing_time': self.performance_stats['total_processing_time']
            },
            'qdrant': qdrant_info,
            'cache': cache_stats,
            'system': {
                'memory_usage_percent': memory.percent,
                'cpu_usage_percent': cpu_percent,
                'memory_available_gb': memory.available / (1024**3)
            }
        }

class QdrantRAGTester:
    """Qdrant RAG 시스템 테스터"""
    
    def __init__(self, rag_system: QdrantRAGSystem):
        self.rag = rag_system
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        logger.info("🧪 Qdrant RAG 시스템 종합 테스트 시작")
        
        # 1. Qdrant 연결 테스트
        self._test_qdrant_connection()
        
        # 2. Ollama 연결 테스트
        self._test_ollama_connection()
        
        # 3. 파일 추가 테스트
        self._test_file_processing()
        
        # 4. 검색 기능 테스트
        self._test_search_functionality()
        
        # 5. 필터링 테스트
        self._test_filtering()
        
        # 6. 파일 관리 테스트
        self._test_file_management()
        
        # 7. 성능 테스트
        self._test_performance()
        
        # 8. 시스템 상태 확인
        self._check_system_status()
    
    def _test_qdrant_connection(self):
        """Qdrant 연결 테스트"""
        logger.info("🔌 Qdrant 연결 테스트...")
        
        try:
            collection_info = self.rag.qdrant_manager.get_collection_info()
            if collection_info:
                logger.info(f"✅ Qdrant 연결 성공: {collection_info['name']}")
                logger.info(f"   포인트 수: {collection_info['points_count']}")
                logger.info(f"   벡터 차원: {collection_info['vector_size']}")
                return True
            else:
                logger.error("❌ Qdrant 연결 실패")
                return False
        except Exception as e:
            logger.error(f"❌ Qdrant 테스트 오류: {e}")
            return False
    
    def _test_ollama_connection(self):
        """Ollama 연결 테스트"""
        logger.info("🤖 Ollama 연결 테스트...")
        
        if self.rag.embedding_client.check_ollama_status():
            # 간단한 임베딩 테스트
            test_embedding = self.rag.embedding_client.embed_text("테스트 문장")
            if test_embedding is not None:
                logger.info(f"✅ 임베딩 테스트 성공 (차원: {len(test_embedding)})")
                return True
            else:
                logger.error("❌ 임베딩 테스트 실패")
                return False
        else:
            logger.error("❌ Ollama 연결 실패")
            return False
    
    def _test_file_processing(self):
        """파일 처리 테스트"""
        logger.info("📁 파일 처리 테스트...")
        
        # 테스트용 임시 파일 생성
        test_files = self._create_test_files()
        
        success_count = 0
        for file_path, content in test_files.items():
            logger.info(f"📄 테스트 파일 처리: {Path(file_path).name}")
            
            if self.rag.add_file(file_path, {'test': True, 'tags': ['test_file']}):
                success_count += 1
                logger.info(f"✅ 파일 처리 성공: {Path(file_path).name}")
            else:
                logger.error(f"❌ 파일 처리 실패: {Path(file_path).name}")
        
        logger.info(f"📊 파일 처리 결과: {success_count}/{len(test_files)}")
        
        # 테스트 파일 정리
        self._cleanup_test_files(test_files.keys())
        
        return success_count == len(test_files)
    
    def _create_test_files(self) -> Dict[str, str]:
        """테스트용 파일 생성"""
        test_files = {}
        temp_dir = tempfile.mkdtemp()
        
        # 한국어 문서
        korean_content = """
# IBM 인공지능 기술 보고서 2024

## 개요
IBM은 인공지능 분야에서 혁신적인 기술을 지속적으로 개발하고 있습니다.
Watson AI 플랫폼을 통해 기업들이 데이터를 활용한 의사결정을 내릴 수 있도록 지원합니다.

## 주요 기술
- 자연어 처리 (NLP)
- 기계학습 (Machine Learning)
- 딥러닝 (Deep Learning)
- 컴퓨터 비전 (Computer Vision)

## DocLing 기술
DocLing은 IBM Research에서 개발한 문서 변환 도구입니다.
HybridChunker의 contextualize 기능으로 문서의 구조적 정보를 보존합니다.
"""
        
        # 영어 문서
        english_content = """
# IBM AI Technology Report 2024

## Overview
IBM continues to develop innovative AI technologies.
The Watson AI platform helps enterprises make data-driven decisions.

## Key Technologies
- Natural Language Processing (NLP)
- Machine Learning
- Deep Learning
- Computer Vision

## DocLing Technology
DocLing is a document conversion tool developed by IBM Research.
The contextualize feature of HybridChunker preserves structural information.
"""
        
        # 기술 문서
        tech_spec = """
# bge-m3 임베딩 모델 명세서

## 모델 정보
- 개발사: BAAI (Beijing Academy of Artificial Intelligence)
- 모델 크기: 2.5GB
- 임베딩 차원: 1024
- 최대 토큰: 8192

## 지원 언어
한국어, 영어, 중국어, 일본어 등 100+ 언어 지원

## 성능
MTEB 벤치마크에서 우수한 성능을 보임
특히 다국어 검색 작업에서 뛰어난 결과
"""
        
        files_to_create = {
            'ibm_ai_report_korean_2024.md': korean_content,
            'ibm_ai_report_english_2024.md': english_content,
            'bge_m3_specification_v1.0.md': tech_spec
        }
        
        for filename, content in files_to_create.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            test_files[file_path] = content
        
        return test_files
    
    def _cleanup_test_files(self, file_paths):
        """테스트 파일 정리"""
        temp_dirs = set()
        for file_path in file_paths:
            temp_dirs.add(os.path.dirname(file_path))
        
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"🧹 테스트 파일 정리: {temp_dir}")
            except Exception as e:
                logger.warning(f"테스트 파일 정리 실패: {e}")
    
    def _test_search_functionality(self):
        """검색 기능 테스트"""
        logger.info("🔍 검색 기능 테스트...")
        
        test_queries = [
            "IBM 인공지능 기술은 무엇인가요?",
            "DocLing의 HybridChunker 기능에 대해 설명해주세요",
            "bge-m3 모델의 성능과 특징은?",
            "자연어 처리 기술",
            "Watson AI platform capabilities"
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 쿼리 {i}: {query}")
            
            results = self.rag.search(query, top_k=3)
            
            if results:
                logger.info(f"📄 {len(results)}개 결과 발견:")
                for j, result in enumerate(results, 1):
                    logger.info(f"  {j}. 유사도: {result.similarity:.3f}")
                    logger.info(f"     파일: {result.metadata.file_name}")
                    logger.info(f"     내용: {result.text[:100]}...")
            else:
                logger.warning("❌ 검색 결과 없음")
    
    def _test_filtering(self):
        """필터링 기능 테스트"""
        logger.info("🎯 필터링 기능 테스트...")
        
        # 파일명 필터
        logger.info("\n📁 파일명 필터 테스트:")
        results = self.rag.search_by_filename("ibm_ai_report_korean_2024.md")
        logger.info(f"한국어 보고서 결과: {len(results)}개")
        
        # 확장자 필터
        logger.info("\n📝 확장자 필터 테스트:")
        results = self.rag.search_by_extension(".md", "IBM")
        logger.info(f"마크다운 파일 검색 결과: {len(results)}개")
        
        # 복합 필터
        logger.info("\n🔗 복합 필터 테스트:")
        current_time = time.time()
        one_hour_ago = current_time - 3600
        results = self.rag.search(
            "기술", 
            top_k=5,
            extension_filter=".md",
            date_range=(one_hour_ago, current_time)
        )
        logger.info(f"최근 1시간 내 마크다운 파일 검색: {len(results)}개")
    
    def _test_file_management(self):
        """파일 관리 기능 테스트"""
        logger.info("📂 파일 관리 기능 테스트...")
        
        # 파일 목록 조회
        file_list = self.rag.get_file_list()
        logger.info(f"📋 저장된 파일 수: {len(file_list)}")
        
        for file_info in file_list[:3]:  # 처음 3개만 표시
            logger.info(f"  📄 {file_info['file_name']}")
            logger.info(f"     크기: {file_info['file_size']} bytes")
            logger.info(f"     청크 수: {file_info['total_chunks']}")
            logger.info(f"     언어: {file_info.get('language', 'auto')}")
    
    def _test_performance(self):
        """성능 테스트"""
        logger.info("⚡ 성능 테스트...")
        
        # 반복 검색으로 캐시 효과 측정
        test_query = "IBM Watson"
        times = []
        
        logger.info(f"🔄 반복 검색 테스트 (쿼리: '{test_query}')")
        for i in range(5):
            start_time = time.time()
            results = self.rag.search(test_query, top_k=5)
            search_time = time.time() - start_time
            times.append(search_time)
            logger.info(f"  검색 {i+1}: {search_time:.3f}초 ({len(results)}개 결과)")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"📊 검색 성능 통계:")
        logger.info(f"  평균: {avg_time:.3f}초")
        logger.info(f"  최소: {min_time:.3f}초")
        logger.info(f"  최대: {max_time:.3f}초")
        
        # 캐시 효과 확인
        cache_stats = self.rag.embedding_client.get_cache_stats()
        logger.info(f"💾 캐시 통계:")
        logger.info(f"  히트율: {cache_stats['hit_rate']:.1%}")
        logger.info(f"  캐시 크기: {cache_stats['cache_size']}")
    
    def _check_system_status(self):
        """시스템 상태 확인"""
        logger.info("📊 시스템 상태 확인...")
        
        status = self.rag.get_system_status()
        
        logger.info(f"🎯 성능 통계:")
        perf = status['performance']
        logger.info(f"  처리된 파일: {perf['files_processed']}")
        logger.info(f"  생성된 청크: {perf['chunks_created']}")
        logger.info(f"  수행된 검색: {perf['searches_performed']}")
        logger.info(f"  평균 검색 시간: {perf['avg_search_time']:.3f}초")
        
        logger.info(f"🗃️ Qdrant 상태:")
        qdrant = status['qdrant']
        logger.info(f"  컬렉션: {qdrant.get('name', 'N/A')}")
        logger.info(f"  포인트 수: {qdrant.get('points_count', 0)}")
        logger.info(f"  상태: {qdrant.get('status', 'N/A')}")
        
        logger.info(f"💻 시스템 리소스:")
        system = status['system']
        logger.info(f"  메모리 사용률: {system['memory_usage_percent']:.1f}%")
        logger.info(f"  CPU 사용률: {system['cpu_usage_percent']:.1f}%")
        logger.info(f"  사용 가능 메모리: {system['memory_available_gb']:.1f}GB")

def setup_and_run_qdrant_demo():
    """Qdrant RAG 데모 설정 및 실행"""
    print("🚀 N100 + DocLing + Ollama bge-m3 + Qdrant RAG 시스템 데모")
    print("=" * 70)
    
    # 설정
    config = QdrantRAGConfig(
        # Qdrant 설정
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="n100_rag_docs",
        
        # 성능 최적화 설정
        max_tokens=512,
        batch_size=3,
        max_parallel=2,
        cache_size=500,
        top_k=10
    )
    
    # RAG 시스템 초기화
    try:
        rag_system = QdrantRAGSystem(config)
        logger.info("✅ Qdrant RAG 시스템 초기화 완료")
    except Exception as e:
        logger.error(f"❌ RAG 시스템 초기화 실패: {e}")
        print("\n💡 설치가 필요한 구성 요소:")
        print("1. Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("2. Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("3. bge-m3: ollama pull bge-m3")
        print("4. Python 패키지: pip install qdrant-client docling")
        return
    
    # 테스터 실행
    tester = QdrantRAGTester(rag_system)
    tester.run_comprehensive_test()
    
    print("\n" + "=" * 70)
    print("🎯 Qdrant + 파일명 메타데이터 시스템 특징:")
    print("• 파일명 자동 메타데이터 추출 (날짜, 버전, 언어, 문서타입)")
    print("• Qdrant 벡터 DB로 빠른 유사도 검색")
    print("• 파일명/확장자/날짜 범위별 필터링")
    print("• N100 CPU 최적화 (배치 처리, 캐싱, 열관리)")
    print("• DocLing contextualize로 문맥 보존")
    
    print("\n💡 주요 사용법:")
    print("• 파일 추가: rag.add_file('/path/to/document.pdf')")
    print("• 검색: rag.search('쿼리', top_k=5)")
    print("• 파일별 검색: rag.search_by_filename('report.pdf')")
    print("• 확장자별 검색: rag.search_by_extension('.md', '내용')")
    print("• 파일 삭제: rag.delete_file('filename.pdf')")
    
    print("\n🔧 실제 사용 예제:")
    print("""
# RAG 시스템 초기화
config = QdrantRAGConfig()
rag = QdrantRAGSystem(config)

# 파일 추가
rag.add_file('reports/IBM_AI_Report_2024.pdf')
rag.add_file('specs/bge-m3_manual_korean.docx')

# 검색
results = rag.search('IBM 인공지능 기술', top_k=5)
for result in results:
    print(f"파일: {result.metadata.file_name}")
    print(f"유사도: {result.similarity:.3f}")
    print(f"내용: {result.text[:200]}...")

# 필터링 검색
pdf_results = rag.search('기술 보고서', extension_filter='.pdf')
recent_results = rag.search('AI', date_range=(yesterday, today))
""")

if __name__ == "__main__":
    # 로그 레벨 설정
    logging.getLogger().setLevel(logging.INFO)
    
    # 데모 실행
    setup_and_run_qdrant_demo()
