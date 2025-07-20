"""
진짜 LangGraph 기반 인텔리전트 멀티 에이전트 토론 시스템
상태 기반 그래프 워크플로우로 구현된 완전한 AI 토론 시스템
"""

import os
import json
import time
import re
import operator
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Annotated, Literal

# 필수 라이브러리 import
try:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
except ImportError as e:
    print(f"❌ 필수 라이브러리가 누락되었습니다: {e}")
    print("설치 명령어:")
    print("pip install openai langchain-openai langgraph")
    exit(1)

# ========================
# 상태 정의 (LangGraph 핵심)
# ========================

@dataclass
class DebateMessage:
    """토론 메시지"""
    speaker: str
    content: str
    timestamp: datetime
    round_num: int
    tokens_used: int = 0
    agent_type: str = ""

class DebateState(dict):
    """LangGraph용 상태 클래스 (TypedDict 스타일)"""
    messages: Annotated[List[DebateMessage], operator.add]
    topic: str
    current_speaker: str
    round_count: int
    max_rounds: int
    is_finished: bool
    topic_analysis: Optional[Dict]
    total_tokens: int
    moderator_analysis: Optional[str]

@dataclass
class AgentProfile:
    """에이전트 프로필"""
    name: str
    role: str
    position: str
    expertise: List[str]
    key_arguments: List[str]
    response_style: str

@dataclass 
class TopicAnalysis:
    """주제 분석 결과"""
    topic: str
    category: str
    stakeholders: List[str]
    key_issues: List[str]
    pro_agent: AgentProfile
    con_agent: AgentProfile

# ========================
# 설정 클래스
# ========================

@dataclass
class LangGraphConfig:
    """LangGraph 설정"""
    openai_api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 500
    max_rounds: int = 3
    enable_checkpoints: bool = True
    save_to_file: bool = True

# ========================
# 주제 분석 시스템
# ========================

class LangGraphTopicAnalyzer:
    """LangGraph용 주제 분석기"""
    
    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            model=config.model,
            temperature=0.5
        )
    
    def analyze_topic(self, topic: str) -> TopicAnalysis:
        """주제 분석 및 에이전트 프로필 생성"""
        print(f"🧠 LangGraph 주제 분석 중: '{topic}'")
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 토론 주제를 전문적으로 분석하는 AI입니다. 
            주어진 주제에 대해 찬반 양측의 이해관계자와 핵심 논점을 분석합니다."""),
            ("user", """다음 토론 주제를 분석하여 JSON 형식으로 응답해주세요: "{topic}"

응답 형식:
{{
  "category": "정책/사회/경제/기술/환경/교육/의료/금융 중 선택",
  "key_issues": ["쟁점1", "쟁점2", "쟁점3", "쟁점4"],
  "pro_stakeholder": "찬성측 이해관계자 (구체적 기관/단체명)",
  "con_stakeholder": "반대측 이해관계자 (구체적 기관/단체명)",
  "pro_arguments": ["찬성 논거1", "찬성 논거2", "찬성 논거3"],
  "con_arguments": ["반대 논거1", "반대 논거2", "반대 논거3"]
}}

한국어로 구체적이고 현실적으로 분석해주세요.""")
        ])
        
        try:
            chain = analysis_prompt | self.llm
            response = chain.invoke({"topic": topic})
            
            # JSON 파싱
            content = response.content
            # JSON 부분만 추출
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
            
            # 에이전트 프로필 생성
            pro_agent = AgentProfile(
                name=analysis_data.get("pro_stakeholder", "찬성 전문가"),
                role=f"{analysis_data.get('category', '정책')} 전문가",
                position="찬성",
                expertise=[analysis_data.get('category', '정책'), "정책 분석"],
                key_arguments=analysis_data.get("pro_arguments", []),
                response_style="적극적이고 미래지향적"
            )
            
            con_agent = AgentProfile(
                name=analysis_data.get("con_stakeholder", "반대 전문가"),
                role=f"{analysis_data.get('category', '정책')} 전문가",
                position="반대", 
                expertise=[analysis_data.get('category', '정책'), "위험 분석"],
                key_arguments=analysis_data.get("con_arguments", []),
                response_style="신중하고 현실적"
            )
            
            return TopicAnalysis(
                topic=topic,
                category=analysis_data.get("category", "정책"),
                stakeholders=[pro_agent.name, con_agent.name],
                key_issues=analysis_data.get("key_issues", []),
                pro_agent=pro_agent,
                con_agent=con_agent
            )
            
        except Exception as e:
            print(f"⚠️ 주제 분석 실패: {e}")
            return self._create_fallback_analysis(topic)
    
    def _create_fallback_analysis(self, topic: str) -> TopicAnalysis:
        """분석 실패 시 기본 에이전트 생성"""
        pro_agent = AgentProfile(
            name="찬성 전문가",
            role="정책 전문가",
            position="찬성",
            expertise=["정책 분석"],
            key_arguments=["필요성 인정", "긍정적 효과"],
            response_style="적극적"
        )
        
        con_agent = AgentProfile(
            name="반대 전문가",
            role="정책 전문가", 
            position="반대",
            expertise=["위험 분석"],
            key_arguments=["부작용 우려", "현실적 한계"],
            response_style="신중함"
        )
        
        return TopicAnalysis(
            topic=topic,
            category="정책",
            stakeholders=["찬성 전문가", "반대 전문가"],
            key_issues=["필요성", "실현가능성", "부작용", "대안"],
            pro_agent=pro_agent,
            con_agent=con_agent
        )

# ========================
# LangGraph 노드 함수들
# ========================

def pro_agent_node(state: DebateState) -> DebateState:
    """찬성 에이전트 노드"""
    print(f"\n✅ **{state['topic_analysis']['pro_agent']['name']}** 발언 중...")
    
    # LLM 초기화
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # 시스템 프롬프트 생성
    agent_info = state["topic_analysis"]["pro_agent"]
    system_prompt = f"""당신은 '{agent_info["name"]}'을 대변하는 {agent_info["role"]}입니다.

토론 주제: {state["topic"]}
당신의 입장: {agent_info["position"]}

핵심 주장:
{chr(10).join(f"- {arg}" for arg in agent_info["key_arguments"])}

응답 스타일: {agent_info["response_style"]}

지침:
1. 당신의 입장을 명확하고 논리적으로 주장하세요
2. 구체적인 근거와 사례를 제시하세요
3. 상대방 주장에 대해 논리적으로 반박하세요
4. 300자 내외로 간결하게 응답하세요
5. 한국어로 응답하세요"""
    
    # 대화 맥락 구성
    context = _build_conversation_context(state["messages"], state["round_count"])
    
    user_message = f"""
토론 주제: {state["topic"]}
현재 라운드: {state["round_count"] + 1}

이전 대화:
{context}

위 맥락을 바탕으로 {agent_info["name"]} 입장에서 논리적으로 응답해주세요.
"""
    
    # AI 응답 생성
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        start_time = time.time()
        response = llm.invoke(messages)
        response_time = time.time() - start_time
        
        # 메시지 생성
        debate_message = DebateMessage(
            speaker=agent_info["name"],
            content=response.content.strip(),
            timestamp=datetime.now(),
            round_num=state["round_count"] + 1,
            tokens_used=getattr(response, 'usage', {}).get('total_tokens', 0),
            agent_type="pro"
        )
        
        print(f"   💬 {response.content.strip()}")
        print(f"   ⏱️ 응답 시간: {response_time:.1f}초")
        
        return {
            "messages": [debate_message],
            "current_speaker": "con",
            "total_tokens": state.get("total_tokens", 0) + debate_message.tokens_used
        }
        
    except Exception as e:
        error_message = DebateMessage(
            speaker=agent_info["name"],
            content=f"[오류] 응답 생성 실패: {str(e)}",
            timestamp=datetime.now(),
            round_num=state["round_count"] + 1,
            agent_type="pro"
        )
        
        return {
            "messages": [error_message],
            "current_speaker": "con",
            "total_tokens": state.get("total_tokens", 0)
        }

def con_agent_node(state: DebateState) -> DebateState:
    """반대 에이전트 노드"""
    print(f"\n❌ **{state['topic_analysis']['con_agent']['name']}** 발언 중...")
    
    # LLM 초기화
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # 시스템 프롬프트 생성
    agent_info = state["topic_analysis"]["con_agent"]
    system_prompt = f"""당신은 '{agent_info["name"]}'을 대변하는 {agent_info["role"]}입니다.

토론 주제: {state["topic"]}
당신의 입장: {agent_info["position"]}

핵심 주장:
{chr(10).join(f"- {arg}" for arg in agent_info["key_arguments"])}

응답 스타일: {agent_info["response_style"]}

지침:
1. 당신의 입장을 명확하고 논리적으로 주장하세요
2. 구체적인 근거와 사례를 제시하세요
3. 상대방 주장에 대해 논리적으로 반박하세요
4. 300자 내외로 간결하게 응답하세요
5. 한국어로 응답하세요"""
    
    # 대화 맥락 구성
    context = _build_conversation_context(state["messages"], state["round_count"])
    
    user_message = f"""
토론 주제: {state["topic"]}
현재 라운드: {state["round_count"] + 1}

이전 대화:
{context}

위 맥락을 바탕으로 {agent_info["name"]} 입장에서 논리적으로 응답해주세요.
"""
    
    # AI 응답 생성
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        start_time = time.time()
        response = llm.invoke(messages)
        response_time = time.time() - start_time
        
        # 메시지 생성
        debate_message = DebateMessage(
            speaker=agent_info["name"],
            content=response.content.strip(),
            timestamp=datetime.now(),
            round_num=state["round_count"] + 1,
            tokens_used=getattr(response, 'usage', {}).get('total_tokens', 0),
            agent_type="con"
        )
        
        print(f"   💬 {response.content.strip()}")
        print(f"   ⏱️ 응답 시간: {response_time:.1f}초")
        
        return {
            "messages": [debate_message],
            "current_speaker": "moderator",
            "total_tokens": state.get("total_tokens", 0) + debate_message.tokens_used
        }
        
    except Exception as e:
        error_message = DebateMessage(
            speaker=agent_info["name"],
            content=f"[오류] 응답 생성 실패: {str(e)}",
            timestamp=datetime.now(),
            round_num=state["round_count"] + 1,
            agent_type="con"
        )
        
        return {
            "messages": [error_message],
            "current_speaker": "moderator",
            "total_tokens": state.get("total_tokens", 0)
        }

def moderator_node(state: DebateState) -> DebateState:
    """사회자 노드 - 라운드 관리 및 흐름 제어"""
    print(f"\n⚖️ **사회자** 라운드 {state['round_count'] + 1} 진행 중...")
    
    # 라운드 증가
    new_round_count = state["round_count"] + 1
    
    # 토론 종료 조건 확인
    if new_round_count >= state["max_rounds"]:
        print(f"   📢 {state['max_rounds']}라운드 완료! 토론을 종료합니다.")
        return {
            "round_count": new_round_count,
            "is_finished": True,
            "current_speaker": "end"
        }
    else:
        print(f"   📢 라운드 {new_round_count} 완료. 다음 라운드를 시작합니다.")
        return {
            "round_count": new_round_count,
            "current_speaker": "pro",  # 다음 라운드는 찬성측부터
            "is_finished": False
        }

def final_analysis_node(state: DebateState) -> DebateState:
    """최종 분석 노드"""
    print(f"\n🎯 **AI 최종 분석** 생성 중...")
    
    # LLM 초기화
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.3  # 분석은 더 객관적으로
    )
    
    # 토론 내용 정리
    conversation_summary = "\n".join([
        f"[{msg.speaker}]: {msg.content}" for msg in state["messages"]
    ])
    
    analysis_prompt = f"""
다음은 '{state["topic"]}'에 대한 토론 내용입니다.

참여자:
- {state["topic_analysis"]["pro_agent"]["name"]} (찬성)
- {state["topic_analysis"]["con_agent"]["name"]} (반대)

토론 내용:
{conversation_summary}

위 토론을 종합적으로 분석하여 다음을 포함한 객관적 평가를 제공해주세요:

1. **핵심 쟁점 정리**: 양측이 집중한 주요 논점들
2. **논리 강도 평가**: 각 측 주장의 설득력과 근거
3. **현실성 검토**: 제시된 방안들의 실현 가능성
4. **균형점 모색**: 양측을 아우르는 해결 방향
5. **종합 평가**: 토론의 전반적 질과 시사점

500자 내외로 전문적이고 균형 잡힌 분석을 제공해주세요.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis_content = response.content.strip()
        
        print(f"   🎯 {analysis_content}")
        
        return {
            "moderator_analysis": analysis_content,
            "total_tokens": state.get("total_tokens", 0) + getattr(response, 'usage', {}).get('total_tokens', 0)
        }
        
    except Exception as e:
        error_analysis = f"[분석 오류] 최종 분석 생성에 실패했습니다: {str(e)}"
        print(f"   ⚠️ {error_analysis}")
        
        return {
            "moderator_analysis": error_analysis,
            "total_tokens": state.get("total_tokens", 0)
        }

# ========================
# 유틸리티 함수들
# ========================

def _build_conversation_context(messages: List[DebateMessage], round_count: int) -> str:
    """대화 맥락 구성"""
    if not messages:
        return "토론 시작 - 첫 발언"
    
    # 최근 6개 메시지만 사용
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    context_parts = []
    
    for msg in recent_messages:
        context_parts.append(f"[Round {msg.round_num}] {msg.speaker}: {msg.content}")
    
    return "\n".join(context_parts)

# ========================
# 라우팅 함수들 (LangGraph 핵심)
# ========================

def should_continue(state: DebateState) -> Literal["pro", "con", "moderator", "final_analysis", "end"]:
    """다음 노드 결정 - LangGraph의 조건부 라우팅"""
    
    if state.get("is_finished", False):
        # 토론이 끝났으면 최종 분석으로
        if state.get("moderator_analysis") is None:
            return "final_analysis"
        else:
            return "end"
    
    current_speaker = state.get("current_speaker", "pro")
    
    if current_speaker == "pro":
        return "pro"
    elif current_speaker == "con": 
        return "con"
    elif current_speaker == "moderator":
        return "moderator"
    else:
        return "end"

# ========================
# LangGraph 워크플로우 생성
# ========================

def create_langgraph_debate_workflow() -> StateGraph:
    """LangGraph 토론 워크플로우 생성"""
    
    # StateGraph 초기화
    workflow = StateGraph(DebateState)
    
    # 노드 추가
    workflow.add_node("pro_agent", pro_agent_node)
    workflow.add_node("con_agent", con_agent_node)
    workflow.add_node("moderator", moderator_node)
    workflow.add_node("final_analysis", final_analysis_node)
    
    # 시작점 설정
    workflow.set_entry_point("pro_agent")
    
    # 조건부 엣지 추가 (LangGraph의 핵심 기능)
    workflow.add_conditional_edges(
        "pro_agent",
        should_continue,
        {
            "con": "con_agent",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "con_agent", 
        should_continue,
        {
            "moderator": "moderator",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "moderator",
        should_continue,
        {
            "pro": "pro_agent",
            "final_analysis": "final_analysis",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "final_analysis",
        should_continue,
        {
            "end": END
        }
    )
    
    return workflow

# ========================
# 메인 LangGraph 토론 시스템
# ========================

class LangGraphDebateSystem:
    """LangGraph 기반 토론 시스템"""
    
    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.topic_analyzer = LangGraphTopicAnalyzer(config)
        
        # 체크포인트 설정 (선택사항)
        self.memory = MemorySaver() if config.enable_checkpoints else None
        
        # 워크플로우 생성
        self.workflow = create_langgraph_debate_workflow()
        
        # 앱 컴파일 (체크포인트 포함)
        if self.memory:
            self.app = self.workflow.compile(checkpointer=self.memory)
        else:
            self.app = self.workflow.compile()
    
    def run_debate(self, topic: str) -> Dict:
        """LangGraph로 토론 실행"""
        print("=" * 80)
        print("🧠 **LangGraph 기반 멀티 에이전트 토론 시스템**")
        print("=" * 80)
        print(f"📋 **토론 주제**: {topic}")
        print("=" * 80)
        
        # 1. 주제 분석
        topic_analysis = self.topic_analyzer.analyze_topic(topic)
        self._print_analysis_results(topic_analysis)
        
        # 2. 초기 상태 설정
        initial_state = {
            "messages": [],
            "topic": topic,
            "current_speaker": "pro",
            "round_count": 0,
            "max_rounds": self.config.max_rounds,
            "is_finished": False,
            "topic_analysis": {
                "pro_agent": {
                    "name": topic_analysis.pro_agent.name,
                    "role": topic_analysis.pro_agent.role,
                    "position": topic_analysis.pro_agent.position,
                    "key_arguments": topic_analysis.pro_agent.key_arguments,
                    "response_style": topic_analysis.pro_agent.response_style
                },
                "con_agent": {
                    "name": topic_analysis.con_agent.name,
                    "role": topic_analysis.con_agent.role,
                    "position": topic_analysis.con_agent.position,
                    "key_arguments": topic_analysis.con_agent.key_arguments,
                    "response_style": topic_analysis.con_agent.response_style
                }
            },
            "total_tokens": 0,
            "moderator_analysis": None
        }
        
        # 3. LangGraph 실행
        try:
            print(f"\n🚀 LangGraph 워크플로우 시작...")
            
            # 체크포인트 설정
            config = {"configurable": {"thread_id": f"debate_{int(time.time())}"}} if self.memory else None
            
            final_state = None
            step_count = 0
            
            # 스트리밍 실행
            for step in self.app.stream(initial_state, config):
                step_count += 1
                print(f"\n🔄 **Step {step_count}**: {list(step.keys())[0]}")
                print("-" * 50)
                
                # 상태 업데이트
                for node_name, node_output in step.items():
                    if node_output:
                        final_state = node_output
                
                # 무한 루프 방지
                if step_count > 20:
                    print("⚠️ 최대 실행 횟수 도달")
                    break
            
            return self._finalize_debate(final_state or initial_state, topic_analysis)
            
        except Exception as e:
            print(f"❌ LangGraph 실행 오류: {e}")
            return {"error": str(e)}
    
    def _print_analysis_results(self, topic_analysis: TopicAnalysis):
        """주제 분석 결과 출력"""
        print(f"\n🎯 **주제 분석 완료**")
        print("=" * 50)
        print(f"📂 분야: {topic_analysis.category}")
        print(f"✅ 찬성측: {topic_analysis.pro_agent.name}")
        print(f"❌ 반대측: {topic_analysis.con_agent.name}")
        print(f"\n🔍 핵심 쟁점:")
        for i, issue in enumerate(topic_analysis.key_issues, 1):
            print(f"   {i}. {issue}")
        print("=" * 50)
    
    def _finalize_debate(self, final_state: Dict, topic_analysis: TopicAnalysis) -> Dict:
        """토론 마무리"""
        print(f"\n" + "=" * 80)
        print("🏁 **LangGraph 토론 완료**")
        print("=" * 80)
        
        # 통계 출력
        messages = final_state.get("messages", [])
        total_tokens = final_state.get("total_tokens", 0)
        
        pro_messages = [m for m in messages if m.agent_type == "pro"]
        con_messages = [m for m in messages if m.agent_type == "con"]
        
        print(f"📊 **최종 통계**:")
        print(f"   • 완료 라운드: {final_state.get('round_count', 0)}/{self.config.max_rounds}")
        print(f"   • 총 메시지: {len(messages)}개")
        print(f"   • 총 토큰: {total_tokens:,} tokens")
        print(f"   • 예상 비용: ${total_tokens * 0.00015:.4f}")
        
        print(f"\n📈 **참여자별 통계**:")
        print(f"✅ {topic_analysis.pro_agent.name}: {len(pro_messages)}회 발언")
        print(f"❌ {topic_analysis.con_agent.name}: {len(con_messages)}회 발언")
        
        if final_state.get("moderator_analysis"):
            print(f"\n🎯 **AI 최종 분석**:")
            print(f"   {final_state['moderator_analysis']}")
        
        print("=" * 80)
        
        # 로그 저장
        if self.config.save_to_file:
            self._save_langgraph_log(final_state, topic_analysis)
        
        return final_state
    
    def _save_langgraph_log(self, final_state: Dict, topic_analysis: TopicAnalysis):
        """LangGraph 토론 로그 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = re.sub(r'[^\w\s-]', '', final_state["topic"].replace(' ', '_'))[:30]
            filename = f"langgraph_debate_{safe_topic}_{timestamp}.json"
            
            # 메시지 직렬화
            serialized_messages = []
            for msg in final_state.get("messages", []):
                serialized_messages.append({
                    "speaker": msg.speaker,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "round_num": msg.round_num,
                    "tokens_used": msg.tokens_used,
                    "agent_type": msg.agent_type
                })
            
            log_data = {
                "metadata": {
                    "system": "LangGraph Multi-Agent Debate",
                    "topic": final_state["topic"],
                    "timestamp": timestamp,
                    "model": self.config.model,
                    "rounds": final_state.get("round_count", 0),
                    "max_rounds": self.config.max_rounds,
                    "checkpoints_enabled": self.config.enable_checkpoints
                },
                "topic_analysis": {
                    "category": topic_analysis.category,
                    "key_issues": topic_analysis.key_issues,
                    "pro_agent": {
                        "name": topic_analysis.pro_agent.name,
                        "role": topic_analysis.pro_agent.role,
                        "key_arguments": topic_analysis.pro_agent.key_arguments
                    },
                    "con_agent": {
                        "name": topic_analysis.con_agent.name,
                        "role": topic_analysis.con_agent.role,
                        "key_arguments": topic_analysis.con_agent.key_arguments
                    }
                },
                "workflow_execution": {
                    "total_tokens": final_state.get("total_tokens", 0),
                    "estimated_cost": final_state.get("total_tokens", 0) * 0.00015,
                    "is_finished": final_state.get("is_finished", False),
                    "moderator_analysis": final_state.get("moderator_analysis")
                },
                "conversation": serialized_messages
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 LangGraph 토론 로그 저장: {filename}")
            
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

# ========================
# 주제 추천 시스템 (기존과 동일)
# ========================

class LangGraphTopicRecommender:
    """LangGraph용 주제 추천 시스템"""
    
    def __init__(self):
        self.topic_categories = {
            "정책": [
                "기본소득 도입의 필요성",
                "국방비 증액은 필요한가",
                "원자력 발전 확대 vs 재생에너지 전환",
                "부동산 보유세 강화 정책",
                "최저임금 대폭 인상의 효과"
            ],
            "사회": [
                "사형제 폐지 vs 유지",
                "동성혼 합법화 논란",
                "청소년 게임 시간 제한 정책",
                "난민 수용 확대 정책",
                "종교 시설 세금 면제 폐지"
            ],
            "기술": [
                "AI 개발 규제의 필요성", 
                "자율주행차 상용화 시기",
                "메타버스 교육 도입",
                "암호화폐 전면 금지 vs 허용",
                "로봇세 도입 논의"
            ],
            "교육": [
                "대학 입시제도 전면 개편",
                "영어 공교육 폐지 논란",
                "AI 시대 코딩 교육 의무화",
                "사교육비 상한제 도입",
                "대학 등록금 무료화 정책"
            ],
            "환경": [
                "플라스틱 사용 전면 금지",
                "탄소세 도입의 효과",
                "원전 vs 재생에너지 우선순위",
                "일회용품 금지 정책 확대",
                "전기차 의무화 시기"
            ],
            "경제": [
                "4일 근무제 도입 효과",
                "대기업 규제 강화 vs 완화",
                "가상화폐 법정화폐 인정",
                "로봇세 vs 기술발전 자유",
                "부의 재분배 정책 강화"
            ],
            "금융": [
                "일반인 주식투자의 필요성",
                "개인투자자 보호 vs 시장 자율성",
                "주식 양도소득세 강화 논란",
                "청소년 금융교육 의무화",
                "가계부채 한도 규제 강화"
            ]
        }
    
    def show_topic_menu(self) -> str:
        """토론 주제 선택 메뉴"""
        print("\n📚 **LangGraph 토론 주제 선택**")
        print("=" * 60)
        
        all_topics = []
        topic_index = 1
        
        for category, topics in self.topic_categories.items():
            print(f"\n🔸 **{category}** 분야:")
            for topic in topics:
                print(f"   {topic_index}. {topic}")
                all_topics.append(topic)
                topic_index += 1
        
        print(f"\n   0. 직접 입력")
        
        while True:
            try:
                choice = input(f"\n주제 선택 (0-{len(all_topics)}): ").strip()
                
                if choice == "0":
                    custom_topic = input("토론 주제를 직접 입력하세요: ").strip()
                    if custom_topic:
                        return custom_topic
                    else:
                        print("⚠️ 주제를 입력해주세요.")
                        continue
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(all_topics):
                    selected_topic = all_topics[choice_num - 1]
                    print(f"✅ 선택된 주제: {selected_topic}")
                    return selected_topic
                else:
                    print(f"⚠️ 1-{len(all_topics)} 범위의 숫자를 입력하세요.")
                    
            except ValueError:
                print("⚠️ 숫자를 입력해주세요.")

# ========================
# 설정 및 실행 함수들
# ========================

def get_langgraph_config() -> LangGraphConfig:
    """LangGraph 설정 수집"""
    print("🔑 **LangGraph 시스템 설정**")
    print("-" * 50)
    
    # OpenAI API 키
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("OpenAI API 키를 입력하세요: ").strip()
        if not api_key:
            print("❌ API 키가 필요합니다!")
            exit(1)
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print(f"✅ 환경변수에서 API 키 로드: {api_key[:8]}...{api_key[-4:]}")
    
    # 모델 선택
    print("\n🧠 AI 모델 선택:")
    models = {
        "1": ("gpt-4o-mini", "빠르고 저렴 (권장)"),
        "2": ("gpt-4o", "최고 품질"),
        "3": ("gpt-3.5-turbo", "가장 저렴")
    }
    
    for key, (model, desc) in models.items():
        print(f"   {key}. {model}: {desc}")
    
    model_choice = input("\n모델 선택 (1-3, 기본값: 1): ").strip()
    selected_model = models.get(model_choice, models["1"])[0]
    
    # 라운드 수
    rounds_input = input("토론 라운드 수 (1-5, 기본값: 3): ").strip()
    try:
        rounds = int(rounds_input)
        rounds = max(1, min(rounds, 5))
    except:
        rounds = 3
    
    # 체크포인트 사용 여부
    checkpoint_input = input("LangGraph 체크포인트 사용? (Y/n, 기본값: Y): ").strip().lower()
    enable_checkpoints = checkpoint_input not in ['n', 'no', '아니요']
    
    print(f"\n✅ **설정 완료**:")
    print(f"   • 모델: {selected_model}")
    print(f"   • 라운드: {rounds}")
    print(f"   • 체크포인트: {'사용' if enable_checkpoints else '미사용'}")
    
    return LangGraphConfig(
        openai_api_key=api_key,
        model=selected_model,
        max_rounds=rounds,
        enable_checkpoints=enable_checkpoints,
        save_to_file=True
    )

def show_langgraph_features():
    """LangGraph 특징 설명"""
    print("\n" + "=" * 70)
    print("🌟 **LangGraph 멀티 에이전트 시스템의 특징**")
    print("=" * 70)
    
    features = {
        "🧠 **상태 기반 워크플로우**": [
            "StateGraph로 토론 상태를 중앙 관리",
            "각 노드가 상태를 업데이트하며 진행",
            "메모리 기반 대화 히스토리 유지"
        ],
        "🔄 **조건부 라우팅**": [
            "should_continue 함수로 동적 흐름 제어", 
            "토론 상황에 따라 다음 노드 자동 결정",
            "복잡한 조건부 로직 구현 가능"
        ],
        "💾 **체크포인트 시스템**": [
            "MemorySaver로 토론 중간 상태 저장",
            "중단 시 이어서 재개 가능",
            "디버깅과 분석에 유용"
        ],
        "🎭 **진정한 멀티 에이전트**": [
            "각 에이전트가 독립된 노드로 동작",
            "병렬 처리 및 비동기 실행 지원",
            "확장 가능한 에이전트 아키텍처"
        ],
        "📊 **고급 모니터링**": [
            "스트리밍 실행으로 실시간 진행 상황 확인",
            "각 단계별 상태 변화 추적",
            "성능 및 토큰 사용량 분석"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   ✓ {item}")

def main():
    """LangGraph 토론 시스템 메인 함수"""
    try:
        print("🧠 **LangGraph 기반 인텔리전트 멀티 에이전트 토론 시스템**")
        print("   진정한 상태 기반 그래프 워크플로우로 구현된 AI 토론")
        print("=" * 70)
        
        # LangGraph 특징 설명
        show_langgraph_features()
        
        # 설정 수집
        config = get_langgraph_config()
        
        # 주제 선택
        topic_recommender = LangGraphTopicRecommender()
        topic = topic_recommender.show_topic_menu()
        
        # LangGraph 시스템 초기화
        print(f"\n🚀 **LangGraph 시스템 초기화 중...**")
        debate_system = LangGraphDebateSystem(config)
        
        # 토론 실행
        final_state = debate_system.run_debate(topic)
        
        if "error" not in final_state:
            total_cost = final_state.get("total_tokens", 0) * 0.00015
            print(f"\n🎉 **LangGraph 토론 완료!**")
            print(f"💰 총 비용: ${total_cost:.4f}")
            print(f"🔄 워크플로우: StateGraph → 조건부 라우팅 → 자동 흐름 제어")
        
        # 추가 옵션
        print(f"\n" + "=" * 60)
        print(f"🎯 **다음 단계 선택**")
        print(f"=" * 60)
        print(f"   1. 새로운 주제로 LangGraph 토론")
        print(f"   2. 같은 주제로 다시 토론")
        print(f"   3. 프로그램 종료")
        
        while True:
            choice = input(f"\n선택하세요 (1-3, 기본값: 3): ").strip()
            
            if choice == "1":
                main()  # 새로운 토론
                break
            elif choice == "2":
                print(f"\n🔄 같은 주제로 LangGraph 토론 재시작...")
                debate_system.run_debate(topic)
                continue
            elif choice == "3" or choice == "":
                print(f"\n👋 LangGraph 토론 시스템을 종료합니다!")
                break
            else:
                print(f"⚠️ 1-3 중에서 선택해주세요.")
        
    except KeyboardInterrupt:
        print("\n⛔ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ LangGraph 시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
