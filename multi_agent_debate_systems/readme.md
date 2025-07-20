# 🤖 Multi-Agent Debate Systems

다양한 접근 방식의 AI 멀티 에이전트 토론 시스템 모음

## 📊 시스템 비교

| 파일명 | 특징 | 난이도 | 추천 용도 |
|--------|------|--------|-----------|
| 01_simple | 데모용, API 키 불필요 | ⭐ | 학습, 테스트 |
| 02_openai_only | 안정적, OpenAI만 | ⭐⭐ | 실용적 사용 |
| 03_intelligent | 동적 분석, 35개 주제 | ⭐⭐⭐ | 범용 토론 |
| 04_langgraph | 최고급, 모든 기능 | ⭐⭐⭐⭐ | 연구, 개발 |

## 🚀 빠른 시작
\```bash
pip install -r requirements.txt
python 01_simple_multi_agent_debate.py
\```


------------------

## LangGraph와 AI Agent의 관계

## 🎭 **AI Agent란?**

### **AI Agent의 정의**
```python
# AI Agent = LLM + Tools + Memory + Decision Making
class AIAgent:
    def __init__(self):
        self.llm = ChatOpenAI()           # 🧠 언어모델
        self.tools = [search, calculator]  # 🔧 도구들  
        self.memory = []                  # 💭 기억
        self.persona = "전문가"            # 🎭 역할
    
    def think_and_act(self, input):
        # 사고 → 도구 사용 → 응답
        return self.llm.invoke(input)
```

### **Agent의 핵심 특징**
- ✅ **자율성**: 스스로 판단하고 행동
- ✅ **목표 지향**: 특정 목표 달성을 위해 행동
- ✅ **도구 사용**: 검색, 계산, API 호출 등
- ✅ **메모리**: 이전 대화/행동 기억
- ✅ **추론**: 상황에 맞는 의사결정

## 🕸️ **LangGraph란?**

### **LangGraph의 정의**
```python
# LangGraph = Agent Orchestration Framework
from langgraph.graph import StateGraph

workflow = StateGraph(State)
workflow.add_node("agent1", agent1_function)
workflow.add_node("agent2", agent2_function)
workflow.add_conditional_edges("agent1", router, {"next": "agent2"})

# 여러 Agent들의 협업을 관리하는 프레임워크
```

### **LangGraph의 핵심 역할**
- 🎼 **오케스트레이션**: 여러 Agent들의 협업 조율
- 🔀 **플로우 제어**: 언제 어떤 Agent가 동작할지 결정
- 📊 **상태 관리**: 모든 Agent가 공유하는 상태 관리
- 🔄 **조건부 라우팅**: 상황에 따른 동적 흐름 제어

## 🤝 **둘의 관계**

### **1. LangGraph = Agent들의 지휘자 🎼**

```python
# 개별 Agent들
doctor_agent = Agent(role="의사", expertise="의료")
government_agent = Agent(role="정부", expertise="정책")
moderator_agent = Agent(role="사회자", expertise="중재")

# LangGraph가 이들을 조율
workflow = StateGraph(DebateState)
workflow.add_node("doctor", doctor_agent.run)
workflow.add_node("government", government_agent.run)  
workflow.add_node("moderator", moderator_agent.run)

# 🎼 지휘: "이제 의사가 발언하고, 다음엔 정부, 그다음엔 사회자"
```

### **2. 계층 구조**

```
🏢 LangGraph (건물 전체)
├── 🎭 Agent 1 (1층 - 의사)
├── 🎭 Agent 2 (2층 - 정부)  
├── 🎭 Agent 3 (3층 - 사회자)
└── 📊 Shared State (엘리베이터 - 정보 공유)
```

## 🔄 **구체적 작동 방식**

### **기존 방식 (Agent만 사용)**
```python
# 순차적, 단순한 Agent 호출
response1 = doctor_agent.chat("의대 정원 확대에 대해 어떻게 생각하세요?")
response2 = government_agent.chat(f"의사가 이렇게 말했는데: {response1}")
response3 = moderator_agent.chat(f"두 의견을 정리하면: {response1}, {response2}")
```

### **LangGraph 방식 (Agent + Orchestration)**
```python
# 상태 기반, 동적 흐름 제어
initial_state = {"messages": [], "topic": "의대 정원 확대"}

for step in app.stream(initial_state):
    # LangGraph가 자동으로:
    # 1. 현재 상태 분석
    # 2. 다음에 누가 말할지 결정
    # 3. 해당 Agent 실행
    # 4. 상태 업데이트
    # 5. 조건 확인 후 다음 단계 결정
```

## 💡 **실제 예시로 이해하기**

### **오케스트라 비유 🎼**

```python
# 개별 연주자들 (Agents)
바이올린_연주자 = Agent("바이올린리스트")
피아노_연주자 = Agent("피아니스트")  
드럼_연주자 = Agent("드러머")

# 지휘자 (LangGraph)
지휘자 = LangGraph()
지휘자.add_musician("바이올린", 바이올린_연주자)
지휘자.add_musician("피아노", 피아노_연주자)
지휘자.add_musician("드럼", 드럼_연주자)

# 🎼 연주 시작
지휘자.start_symphony():
    # 1악장: 바이올린 솔로
    # 2악장: 피아노 + 바이올린 듀엣  
    # 3악장: 전체 합주
    # 상황에 따라 동적으로 지휘!
```

## 🎯 **핵심 포인트**

### **Agent 없이 LangGraph?** ❌
```python
# 의미없음 - 지휘할 연주자가 없음
workflow = StateGraph()
# 빈 오케스트라... 😅
```

### **LangGraph 없이 Agent들?** 😕  
```python
# 가능하지만 제한적
agent1.chat() → agent2.chat() → agent3.chat()
# 단순 순차 실행, 복잡한 협업 어려움
```

### **Agent + LangGraph** ✨
```python
# 진정한 멀티 에이전트 시스템!
# 각자의 전문성 + 체계적 협업 = 시너지 극대화
```

## 📊 **정리**

| 구분 | Agent | LangGraph |
|------|-------|-----------|
| **역할** | 개별 전문가 | 협업 관리자 |
| **기능** | 사고, 추론, 도구 사용 | 흐름 제어, 상태 관리 |
| **범위** | 단일 작업 | 복합 워크플로우 |
| **비유** | 연주자 | 지휘자 |

**결론**: LangGraph는 여러 AI Agent들이 **체계적으로 협업**할 수 있게 해주는 **오케스트레이션 프레임워크**입니다! 🎼✨
