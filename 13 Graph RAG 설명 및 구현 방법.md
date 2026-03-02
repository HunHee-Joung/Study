# Dify 워크플로우 - Graph RAG 검색 노드 추가 가이드

> **대상**: BNK 부산은행 IT기획부 AI 인프라 담당자  
> **환경**: Dify + Neo4j + Qwen3-30B (온프레미스)  
> **목적**: 기존 Vector RAG 워크플로우에 그래프 검색 노드를 통합하여 관계형 지식 검색 구현

---

## 목차

1. [전체 아키텍처](#1-전체-아키텍처)
2. [사전 준비](#2-사전-준비)
3. [Neo4j 설치 및 설정](#3-neo4j-설치-및-설정)
4. [Graph API 서버 구축](#4-graph-api-서버-구축)
5. [Dify 워크플로우 구성](#5-dify-워크플로우-구성)
6. [방법별 상세 구현](#6-방법별-상세-구현)
   - [방법 1: HTTP Request 노드 (권장)](#방법-1-http-request-노드-권장)
   - [방법 2: Code 노드](#방법-2-code-노드)
   - [방법 3: Custom Tool 플러그인](#방법-3-custom-tool-플러그인)
7. [그래프 데이터 구축 파이프라인](#7-그래프-데이터-구축-파이프라인)
8. [테스트 및 검증](#8-테스트-및-검증)
9. [트러블슈팅](#9-트러블슈팅)
10. [구현 로드맵](#10-구현-로드맵)

---

## 1. 전체 아키텍처

### 기존 Vector RAG vs Graph RAG 통합 흐름

```
[기존 Dify RAG 흐름]
사용자 입력 ──→ 지식검색(Vector DB) ──→ LLM 답변 생성 ──→ 출력

[Graph RAG 통합 흐름]
                          ┌──→ 그래프 검색 (Neo4j) ──┐
사용자 입력 ──→ 엔티티 추출 ─┤                          ├──→ 컨텍스트 병합 ──→ LLM ──→ 출력
                          └──→ 벡터 검색 (기존 RAG) ──┘
```

### 시스템 구성도

```
┌─────────────────────────────────────────────────────────────┐
│                        Dify Platform                        │
│                                                             │
│  [입력 노드] → [LLM: 엔티티 추출] → [HTTP Request: 그래프 검색] │
│                                  → [지식검색: 벡터 검색]      │
│                                  → [LLM: 최종 답변 생성]      │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP POST
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              Graph API Server (FastAPI :8001)                │
│                                                             │
│   /entity/extract  →  Ollama Qwen3-30B (:11434)             │
│   /graph/search    →  Neo4j (:7687)                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 사전 준비

### 필요 소프트웨어

| 소프트웨어 | 버전 | 용도 |
|-----------|------|------|
| Docker | 20.x 이상 | Neo4j 컨테이너 실행 |
| Python | 3.10 이상 | Graph API 서버 |
| Ollama | 최신 | Qwen3-30B 로컬 LLM |
| Neo4j | 5.x | 그래프 데이터베이스 |

### 포트 정보

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Neo4j Browser | 7474 | 웹 UI |
| Neo4j Bolt | 7687 | 드라이버 연결 |
| Graph API | 8001 | FastAPI 서버 |
| Ollama | 11434 | LLM API |
| Dify | 80/443 | 플랫폼 |

---

## 3. Neo4j 설치 및 설정

### Docker Compose로 설치

```yaml
# docker-compose-neo4j.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15.0
    container_name: neo4j_graphrag
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/bnk_password_2024
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
      - NEO4J_dbms_memory_heap_max__size=4G
      - NEO4J_dbms_memory_pagecache_size=2G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
```

```bash
# 실행
docker-compose -f docker-compose-neo4j.yml up -d

# 상태 확인
docker logs neo4j_graphrag

# 브라우저 접속: http://localhost:7474
# ID: neo4j / PW: bnk_password_2024
```

### 은행 도메인 스키마 초기화

Neo4j Browser 또는 cypher-shell에서 실행:

```cypher
-- 노드 인덱스 생성
CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name);
CREATE INDEX entity_type_index IF NOT EXISTS FOR (n:Entity) ON (n.type);

-- 제약 조건
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
  FOR (n:Entity) REQUIRE n.id IS UNIQUE;

-- 은행 도메인 샘플 데이터
CREATE (a:Entity {id: "E001", name: "기업여신규정", type: "규정", description: "기업 대출 심사 기준 규정"})
CREATE (b:Entity {id: "E002", name: "여신심사팀", type: "부서", description: "여신 심사 담당 부서"})
CREATE (c:Entity {id: "E003", name: "신용등급평가", type: "프로세스", description: "차주 신용등급 평가 절차"})
CREATE (a)-[:APPLIES_TO {description: "심사 기준 적용"}]->(b)
CREATE (b)-[:EXECUTES]->(c)
CREATE (c)-[:BASED_ON]->(a);
```

---

## 4. Graph API 서버 구축

### 프로젝트 구조

```
graph_api/
├── main.py          # FastAPI 앱
├── graph_search.py  # Neo4j 검색 로직
├── entity_extract.py # 엔티티 추출 (Qwen3)
├── requirements.txt
└── Dockerfile
```

### requirements.txt

```
fastapi==0.109.0
uvicorn==0.27.0
neo4j==5.17.0
httpx==0.26.0
pydantic==2.5.0
python-dotenv==1.0.0
```

### main.py

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import re
import os
from neo4j import GraphDatabase

app = FastAPI(title="Graph RAG API", version="1.0.0")

# CORS 설정 (Dify에서 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j 연결
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "bnk_password_2024")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ─────────────────────────────────────────────
# 요청/응답 모델
# ─────────────────────────────────────────────

class EntityExtractRequest(BaseModel):
    question: str

class GraphSearchRequest(BaseModel):
    question: str
    entities: list[str]
    depth: int = 2
    limit: int = 20

class GraphSearchResponse(BaseModel):
    graph_context: str
    entity_count: int
    relation_count: int
    raw_results: Optional[list] = None


# ─────────────────────────────────────────────
# 엔티티 추출 API
# ─────────────────────────────────────────────

@app.post("/entity/extract")
async def extract_entities(req: EntityExtractRequest):
    """질문에서 핵심 엔티티 추출 (Qwen3 활용)"""
    prompt = f"""다음 질문에서 검색에 필요한 핵심 엔티티를 추출하세요.
엔티티 유형: 기업명, 상품명, 규정명, 부서명, 프로세스명, 법령명

반드시 JSON 배열 형식으로만 출력하세요. 다른 텍스트 없이 배열만 출력.

질문: {req.question}

출력 예시: ["기업여신규정", "여신심사팀", "신용등급"]
"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": "qwen3:30b", "prompt": prompt, "stream": False}
            )
            text = response.json()["response"].strip()
            # JSON 배열 추출
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            entities = json.loads(match.group()) if match else []
            return {"entities": entities, "count": len(entities)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"엔티티 추출 실패: {str(e)}")


# ─────────────────────────────────────────────
# 그래프 검색 API
# ─────────────────────────────────────────────

@app.post("/graph/search", response_model=GraphSearchResponse)
async def graph_search(req: GraphSearchRequest):
    """Neo4j 그래프에서 엔티티 관계 탐색"""
    all_results = []

    with driver.session() as session:
        for entity in req.entities:
            cypher = f"""
            MATCH (n)
            WHERE n.name CONTAINS $entity
            MATCH path = (n)-[r*1..{req.depth}]-(related)
            RETURN
                n.name      AS source,
                n.type      AS source_type,
                [rel IN relationships(path) | type(rel)][-1] AS relation,
                related.name        AS target,
                related.type        AS target_type,
                related.description AS description
            LIMIT {req.limit}
            """
            result = session.run(cypher, entity=entity)
            for record in result:
                all_results.append({
                    "source": record["source"],
                    "source_type": record["source_type"],
                    "relation": record["relation"],
                    "target": record["target"],
                    "target_type": record["target_type"],
                    "description": record["description"]
                })

    # 중복 제거
    seen = set()
    unique_results = []
    for r in all_results:
        key = f"{r['source']}_{r['relation']}_{r['target']}"
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    # 엔티티/관계 집계
    entities_found = set()
    for r in unique_results:
        entities_found.add(r['source'])
        entities_found.add(r['target'])

    graph_context = _format_graph_context(unique_results)

    return GraphSearchResponse(
        graph_context=graph_context,
        entity_count=len(entities_found),
        relation_count=len(unique_results),
        raw_results=unique_results
    )


def _format_graph_context(results: list) -> str:
    if not results:
        return "관련 그래프 정보를 찾을 수 없습니다."

    lines = ["[그래프 검색 결과 - 엔티티 관계 정보]"]
    lines.append(f"총 {len(results)}개의 관계 발견\n")

    for r in results:
        src_type = f"({r['source_type']})" if r.get('source_type') else ""
        tgt_type = f"({r['target_type']})" if r.get('target_type') else ""
        line = f"• {r['source']}{src_type} ─[{r['relation']}]→ {r['target']}{tgt_type}"
        lines.append(line)
        if r.get('description'):
            lines.append(f"  └ {r['description']}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 헬스체크
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        return {"status": "ok", "neo4j": "connected"}
    except Exception as e:
        return {"status": "error", "neo4j": str(e)}
```

### 서버 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=bnk_password_2024
export OLLAMA_URL=http://localhost:11434

# 실행
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# 서버 확인
curl http://localhost:8001/health
```

---

## 5. Dify 워크플로우 구성

### 노드 배치 순서

```
[시작] 
  │
  ▼
[LLM 노드 ①] ── 엔티티 추출
  │  입력: {{#sys.query#}}
  │  출력: entities_text (JSON 배열 문자열)
  │
  ├────────────────────────┐
  ▼                        ▼
[HTTP Request 노드 ②]  [지식검색 노드 ③]
그래프 검색                벡터 검색
POST /graph/search        기존 Vector DB
출력: graph_context        출력: vector_context
  │                        │
  └──────────┬─────────────┘
             ▼
        [LLM 노드 ④] ── 최종 답변 생성
             │
             ▼
           [종료]
```

### 노드별 상세 설정

#### ① 엔티티 추출 LLM 노드

| 항목 | 설정값 |
|------|--------|
| 노드명 | `엔티티_추출` |
| 모델 | qwen3:30b (Ollama) |
| 출력 변수명 | `entities_text` |

**시스템 프롬프트:**
```
당신은 텍스트에서 핵심 엔티티를 추출하는 전문가입니다.
반드시 JSON 배열 형식으로만 응답하세요.
```

**사용자 프롬프트:**
```
다음 질문에서 은행 업무 관련 핵심 엔티티를 추출하세요.
(기업명, 규정명, 상품명, 부서명, 프로세스명 등)

반드시 JSON 배열로만 출력하세요.
예: ["기업여신규정", "여신심사팀"]

질문: {{#sys.query#}}
```

---

#### ② 그래프 검색 HTTP Request 노드

| 항목 | 설정값 |
|------|--------|
| 노드명 | `그래프_검색` |
| Method | POST |
| URL | `http://localhost:8001/graph/search` |

**Headers:**
```json
{
  "Content-Type": "application/json"
}
```

**Body (Raw JSON):**
```json
{
  "question": "{{#sys.query#}}",
  "entities": {{#엔티티_추출.entities_text#}},
  "depth": 2,
  "limit": 20
}
```

**출력 변수 설정:**
```
변수명: graph_context
경로:   body.graph_context

변수명: relation_count
경로:   body.relation_count
```

---

#### ③ 지식검색 노드 (기존 유지)

| 항목 | 설정값 |
|------|--------|
| 노드명 | `벡터_검색` |
| 지식베이스 | 기존 Vector DB 선택 |
| Top K | 5 |
| Score 임계값 | 0.5 |

---

#### ④ 최종 답변 생성 LLM 노드

| 항목 | 설정값 |
|------|--------|
| 노드명 | `최종_답변` |
| 모델 | qwen3:30b |

**시스템 프롬프트:**
```
당신은 BNK 부산은행의 AI 어시스턴트입니다.
제공된 두 가지 정보를 종합하여 정확하고 유용한 답변을 제공하세요.

답변 원칙:
1. 그래프 정보로 엔티티 간 관계와 구조를 파악하세요.
2. 문서 정보로 구체적인 내용을 보완하세요.
3. 두 정보가 상충할 경우 문서 정보를 우선하세요.
4. 불확실한 내용은 명확히 표시하세요.
```

**사용자 프롬프트:**
```
## 그래프 검색 결과 (관계/구조 정보)
{{#그래프_검색.graph_context#}}

## 문서 검색 결과 (상세 내용)
{{#벡터_검색.result#}}

## 질문
{{#sys.query#}}

위 정보를 바탕으로 질문에 답변하세요.
```

---

## 6. 방법별 상세 구현

### 방법 1: HTTP Request 노드 (권장)

**장점**: 설정 간편, 별도 코드 불필요, 시각적 워크플로우 유지  
**단점**: 외부 API 서버 운영 필요

**흐름:**
```
Dify HTTP Request 노드 → FastAPI 서버(:8001) → Neo4j(:7687)
```

위 [섹션 4](#4-graph-api-서버-구축)의 FastAPI 서버를 구축한 후 [섹션 5](#5-dify-워크플로우-구성)의 HTTP Request 노드 설정을 따르면 됩니다.

---

### 방법 2: Code 노드

**장점**: 외부 서버 불필요, Dify 내에서 완결  
**단점**: Dify Code 노드 패키지 제한 (neo4j 드라이버 사용 불가 → HTTP로 우회)

Dify의 **Code 노드** → Python 선택 후 아래 코드 입력:

```python
def main(query: str, entities_text: str) -> dict:
    import json
    import urllib.request
    import urllib.error

    # 엔티티 파싱
    try:
        entities = json.loads(entities_text)
    except:
        entities = []

    if not entities:
        return {"graph_context": "추출된 엔티티가 없습니다.", "success": False}

    # Graph API 호출
    payload = json.dumps({
        "question": query,
        "entities": entities,
        "depth": 2,
        "limit": 20
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            "http://localhost:8001/graph/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        return {
            "graph_context": result.get("graph_context", "결과 없음"),
            "relation_count": result.get("relation_count", 0),
            "success": True
        }

    except urllib.error.URLError as e:
        return {
            "graph_context": f"그래프 검색 서버 연결 실패: {str(e)}",
            "success": False
        }
    except Exception as e:
        return {
            "graph_context": f"그래프 검색 오류: {str(e)}",
            "success": False
        }
```

**노드 입력 변수 설정:**

| 변수명 | 값 |
|--------|-----|
| `query` | `{{#sys.query#}}` |
| `entities_text` | `{{#엔티티_추출.entities_text#}}` |

**노드 출력 변수:**
- `graph_context` (string)
- `relation_count` (number)
- `success` (boolean)

---

### 방법 3: Custom Tool 플러그인

**장점**: 에이전트 모드에서 LLM이 자동으로 도구 선택  
**단점**: 에이전트 설정 필요, 워크플로우보다 제어가 어려움

#### OpenAPI 스펙 파일 작성

```yaml
# graph_search_openapi.yaml
openapi: 3.0.0
info:
  title: Graph Search API
  description: Neo4j 지식 그래프 검색 API
  version: 1.0.0
servers:
  - url: http://localhost:8001

paths:
  /graph/search:
    post:
      operationId: searchKnowledgeGraph
      summary: 지식 그래프에서 엔티티 관계 검색
      description: 은행 업무 관련 엔티티와 관계를 그래프에서 탐색합니다
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [question, entities]
              properties:
                question:
                  type: string
                  description: 사용자 질문
                entities:
                  type: array
                  items:
                    type: string
                  description: 검색할 엔티티 목록
                depth:
                  type: integer
                  default: 2
                  description: 그래프 탐색 깊이 (1~3)
      responses:
        '200':
          description: 검색 결과
          content:
            application/json:
              schema:
                type: object
                properties:
                  graph_context:
                    type: string
                    description: 포맷된 그래프 검색 결과
                  entity_count:
                    type: integer
                  relation_count:
                    type: integer
```

#### Dify Custom Tool 등록 방법

```
1. Dify 관리자 페이지 접속
2. [Tools] → [Custom Tools] → [Create Custom Tool] 클릭
3. 위 YAML 내용 붙여넣기
4. [Test] 탭에서 동작 확인
5. 워크플로우 또는 에이전트에서 해당 도구 선택 가능
```

---

## 7. 그래프 데이터 구축 파이프라인

### 문서 → 그래프 변환 스크립트

```python
# ingest_to_graph.py
import json
import httpx
from neo4j import GraphDatabase
import asyncio

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "bnk_password_2024")
OLLAMA_URL = "http://localhost:11434"

EXTRACTION_PROMPT = """
다음 문서에서 엔티티와 관계를 추출하세요.

엔티티 유형: 기업, 상품, 규정, 부서, 담당자, 프로세스, 법령, 금액, 기간
관계 유형: 신청, 심사, 적용, 소속, 참조, 개정, 승인, 반려, 위임, 준수

출력은 반드시 아래 JSON 형식으로만 작성하세요:
{{
  "entities": [
    {{"id": "E1", "name": "이름", "type": "유형", "description": "설명"}}
  ],
  "relations": [
    {{"source_id": "E1", "target_id": "E2", "type": "관계유형", "description": "설명"}}
  ]
}}

문서:
{text}
"""

async def extract_entities_relations(text: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "qwen3:30b",
                "prompt": EXTRACTION_PROMPT.format(text=text[:3000]),
                "stream": False
            }
        )
    raw = response.json()["response"]
    
    # JSON 추출
    import re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"entities": [], "relations": []}


def ingest_to_neo4j(doc_id: str, extracted: dict):
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    with driver.session() as session:
        # 엔티티 적재
        for entity in extracted.get("entities", []):
            session.run("""
                MERGE (n:Entity {name: $name})
                SET n.type = $type,
                    n.description = $desc,
                    n.doc_id = $doc_id,
                    n.updated_at = datetime()
            """,
            name=entity["name"],
            type=entity.get("type", "Unknown"),
            desc=entity.get("description", ""),
            doc_id=doc_id
            )
        
        # 관계 매핑 (id → name)
        id_to_name = {e["id"]: e["name"] for e in extracted.get("entities", [])}
        
        # 관계 적재
        for rel in extracted.get("relations", []):
            src_name = id_to_name.get(rel["source_id"])
            tgt_name = id_to_name.get(rel["target_id"])
            if not src_name or not tgt_name:
                continue
            
            rel_type = rel["type"].replace(" ", "_").upper()
            session.run(f"""
                MATCH (a:Entity {{name: $src}})
                MATCH (b:Entity {{name: $tgt}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.description = $desc,
                    r.doc_id = $doc_id
            """,
            src=src_name,
            tgt=tgt_name,
            desc=rel.get("description", ""),
            doc_id=doc_id
            )
    
    driver.close()
    print(f"[완료] {doc_id}: 엔티티 {len(extracted.get('entities',[]))}개, "
          f"관계 {len(extracted.get('relations',[]))}개 적재")


async def process_document(doc_id: str, text: str):
    print(f"[처리중] {doc_id}...")
    extracted = await extract_entities_relations(text)
    ingest_to_neo4j(doc_id, extracted)


# 실행 예시
if __name__ == "__main__":
    sample_docs = [
        ("DOC001", "기업여신규정에 따라 여신심사팀은 차주의 신용등급을 평가해야 합니다..."),
        ("DOC002", "무역금융 LC 발행 시 국제상공회의소 UCP600 규정을 준수해야 합니다..."),
    ]
    
    async def main():
        for doc_id, text in sample_docs:
            await process_document(doc_id, text)
    
    asyncio.run(main())
```

---

## 8. 테스트 및 검증

### API 단위 테스트

```bash
# 1. 헬스체크
curl http://localhost:8001/health

# 2. 엔티티 추출 테스트
curl -X POST http://localhost:8001/entity/extract \
  -H "Content-Type: application/json" \
  -d '{"question": "기업여신 심사 시 신용등급 평가 기준은?"}'

# 3. 그래프 검색 테스트
curl -X POST http://localhost:8001/graph/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "신용등급 평가 기준은?",
    "entities": ["기업여신규정", "신용등급평가"],
    "depth": 2
  }'
```

### 기대 출력 예시

```json
{
  "graph_context": "[그래프 검색 결과 - 엔티티 관계 정보]\n총 3개의 관계 발견\n\n• 기업여신규정(규정) ─[APPLIES_TO]→ 여신심사팀(부서)\n  └ 심사 기준 적용\n• 여신심사팀(부서) ─[EXECUTES]→ 신용등급평가(프로세스)\n• 신용등급평가(프로세스) ─[BASED_ON]→ 기업여신규정(규정)",
  "entity_count": 3,
  "relation_count": 3
}
```

### Dify 워크플로우 테스트

```
테스트 질문 1: "기업여신 심사 절차를 설명해줘"
  → 그래프: 여신심사팀 → 심사프로세스 → 규정 관계 탐색
  → 벡터: 관련 문서 청크 검색
  → 통합 답변 생성 확인

테스트 질문 2: "UCP600이 적용되는 상품은?"
  → 그래프: UCP600 → 적용상품 관계 탐색
  → 멀티홉 추론 가능 여부 확인

테스트 질문 3: "여신심사팀의 담당 규정 목록은?"
  → 그래프: 여신심사팀 → 관련규정 역방향 탐색
  → 관계 기반 집계 확인
```

### 성능 비교 체크리스트

| 항목 | Vector Only | Graph + Vector | 개선 여부 |
|------|------------|----------------|----------|
| 단순 내용 검색 | ✅ | ✅ | 동일 |
| 엔티티 관계 파악 | ❌ | ✅ | 개선 |
| 멀티-홉 추론 | ❌ | ✅ | 개선 |
| 전체 구조 파악 | △ | ✅ | 개선 |
| 응답 속도 | 빠름 | 약간 느림 | 허용 범위 내 |

---

## 9. 트러블슈팅

### Neo4j 연결 실패

```bash
# 컨테이너 상태 확인
docker ps | grep neo4j

# 로그 확인
docker logs neo4j_graphrag --tail 50

# Bolt 포트 확인
netstat -tlnp | grep 7687
```

### 엔티티 추출 결과 없음

- Ollama Qwen3 모델 로드 확인: `ollama ps`
- 프롬프트에서 "반드시 JSON 배열로만" 강조 문구 추가
- 모델 응답 로그 확인 후 정규식 패턴 조정

### Dify HTTP Request 타임아웃

```
설정 위치: HTTP Request 노드 → Advanced Settings → Timeout
권장값: 30000 (30초)
```

### 그래프 검색 결과가 빈 배열

```cypher
-- Neo4j에서 직접 확인
MATCH (n:Entity) RETURN n LIMIT 10;

-- 엔티티명 부분 일치 확인
MATCH (n:Entity) WHERE n.name CONTAINS "여신" RETURN n.name;
```

---

## 10. 구현 로드맵

### Phase 1 - 기반 구축 (1주차)

- [ ] Neo4j Docker 설치 및 스키마 초기화
- [ ] FastAPI Graph API 서버 구축
- [ ] `/health`, `/graph/search` 엔드포인트 동작 확인
- [ ] 샘플 데이터 50건 수동 적재

### Phase 2 - Dify 통합 (2주차)

- [ ] Dify 워크플로우에 HTTP Request 노드 추가
- [ ] 엔티티 추출 → 그래프 검색 → 벡터 검색 → 답변 생성 흐름 구성
- [ ] 10개 테스트 질문으로 품질 검증
- [ ] Vector Only vs Graph + Vector 답변 품질 비교

### Phase 3 - 데이터 확장 (3주차)

- [ ] 기존 문서 파이프라인에 `ingest_to_graph.py` 연동
- [ ] 무역금융 / 여신 / 규정 문서 자동 적재
- [ ] 그래프 노드 500개 이상 목표
- [ ] 탐색 깊이 및 결과 품질 최적화

### Phase 4 - 고도화 (4주차)

- [ ] 커뮤니티 감지 (Neo4j GDS - Leiden Algorithm) 적용
- [ ] 그래프 시각화 대시보드 구축
- [ ] 에이전트 모드 Custom Tool 전환 검토
- [ ] 성능 모니터링 및 운영 문서화

---

## 참고

| 자료 | URL |
|------|-----|
| Neo4j Cypher 문서 | https://neo4j.com/docs/cypher-manual/ |
| Microsoft GraphRAG | https://github.com/microsoft/graphrag |
| Dify Workflow 문서 | https://docs.dify.ai/guides/workflow |
| Neo4j Python Driver | https://neo4j.com/docs/python-manual/ |

---

*문서 작성: BNK 부산은행 IT기획부 AI인프라팀*  
*최종 업데이트: 2024년*
