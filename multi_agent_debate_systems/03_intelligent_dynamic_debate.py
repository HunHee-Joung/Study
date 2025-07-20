"""
인텔리전트 동적 멀티 에이전트 토론 시스템
토론 주제에 따라 에이전트 역할과 논점을 자동으로 생성하는 AI 시스템
"""

import os
import json
import time
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime

# 필수 라이브러리 import
try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI 라이브러리가 필요합니다: pip install openai")
    exit(1)

# ========================
# 설정 클래스들
# ========================

@dataclass
class APIConfig:
    """API 설정"""
    openai_api_key: str = "openai_api"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 500
    timeout: int = 30

@dataclass
class DebateConfig:
    """토론 설정"""
    max_rounds: int = 3
    response_delay: float = 1.5
    save_to_file: bool = True
    include_analysis: bool = True

@dataclass
class AgentProfile:
    """에이전트 프로필"""
    name: str
    role: str
    position: str  # 찬성/반대
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

@dataclass
class Message:
    """메시지 클래스"""
    speaker: str
    content: str
    timestamp: datetime
    round_num: int
    tokens_used: int = 0
    response_time: float = 0.0

@dataclass
class DebateState:
    """토론 상태"""
    messages: List[Message] = field(default_factory=list)
    current_speaker: str = ""
    round_count: int = 0
    topic: str = ""
    topic_analysis: Optional[TopicAnalysis] = None
    is_finished: bool = False
    total_tokens: int = 0
    total_cost: float = 0.0

# ========================
# 주제 분석 및 에이전트 생성 시스템
# ========================

class TopicAnalyzer:
    """토론 주제 분석 및 에이전트 생성"""
    
    def __init__(self, client: OpenAI, config: APIConfig):
        self.client = client
        self.config = config
    
    def analyze_topic(self, topic: str) -> TopicAnalysis:
        """주제를 분석하고 에이전트 프로필 생성"""
        print(f"🧠 주제 분석 중: '{topic}'")
        
        # 1단계: 주제 분석
        analysis_prompt = f"""
다음 토론 주제를 분석해주세요: "{topic}"

다음 형식으로 분석 결과를 제공해주세요:

CATEGORY: [정책/사회/경제/기술/환경/교육/의료/기타 중 선택]

STAKEHOLDERS: [이 주제와 관련된 주요 이해관계자들을 2개 그룹으로 나누어 명시]
- 찬성측: [구체적인 기관/단체/직업군]
- 반대측: [구체적인 기관/단체/직업군]

KEY_ISSUES: [이 주제의 핵심 쟁점들을 4-5개로 정리]
- [쟁점1]
- [쟁점2]
- [쟁점3]
- [쟁점4]

PRO_ARGUMENTS: [찬성 측의 주요 논거 4-5개]
- [논거1]
- [논거2]
- [논거3]
- [논거4]

CON_ARGUMENTS: [반대 측의 주요 논거 4-5개]
- [논거1]
- [논거2]
- [논거3]
- [논거4]

한국어로 구체적이고 현실적으로 분석해주세요.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "당신은 토론 주제를 전문적으로 분석하는 정책 분석가입니다. 다양한 분야의 토론 주제를 객관적이고 균형 있게 분석할 수 있습니다."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.5,
                max_tokens=800,
                timeout=self.config.timeout
            )
            
            analysis_text = response.choices[0].message.content
            print(f"✅ 주제 분석 완료 ({response.usage.total_tokens} tokens)")
            
            # 2단계: 분석 결과 파싱 및 에이전트 생성
            return self._parse_analysis_and_create_agents(topic, analysis_text)
            
        except Exception as e:
            print(f"❌ 주제 분석 실패: {e}")
            # 기본 에이전트로 폴백
            return self._create_default_agents(topic)
    
    def _parse_analysis_and_create_agents(self, topic: str, analysis_text: str) -> TopicAnalysis:
        """분석 결과를 파싱하고 에이전트 생성"""
        
        # 정규식으로 파싱
        def extract_section(text: str, section: str) -> List[str]:
            pattern = f"{section}:(.*?)(?=\n[A-Z_]+:|$)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # - 로 시작하는 항목들 추출
                items = [line.strip("- ").strip() for line in content.split('\n') if line.strip().startswith('-')]
                return items
            return []
        
        def extract_single(text: str, section: str) -> str:
            pattern = rf"{section}:\s*\[?([^\[\n]+)\]?"
            match = re.search(pattern, text)
            return match.group(1).strip() if match else ""
        
        category = extract_single(analysis_text, "CATEGORY")
        key_issues = extract_section(analysis_text, "KEY_ISSUES")
        pro_arguments = extract_section(analysis_text, "PRO_ARGUMENTS") 
        con_arguments = extract_section(analysis_text, "CON_ARGUMENTS")
        
        # 이해관계자 추출
        stakeholders_text = re.search(r"STAKEHOLDERS:(.*?)(?=\nKEY_ISSUES:|$)", analysis_text, re.DOTALL)
        pro_stakeholder = "찬성 측"
        con_stakeholder = "반대 측"
        
        if stakeholders_text:
            stakeholder_content = stakeholders_text.group(1)
            pro_match = re.search(r"찬성측?[:\s]*([^\n]+)", stakeholder_content)
            con_match = re.search(r"반대측?[:\s]*([^\n]+)", stakeholder_content)
            
            if pro_match:
                pro_stakeholder = pro_match.group(1).strip("[] ")
            if con_match:
                con_stakeholder = con_match.group(1).strip("[] ")
        
        # 에이전트 프로필 생성
        pro_agent = AgentProfile(
            name=pro_stakeholder,
            role=f"{category} 분야 전문가",
            position="찬성",
            expertise=[category, "정책 분석", "현황 파악"],
            key_arguments=pro_arguments,
            response_style="적극적이고 미래지향적"
        )
        
        con_agent = AgentProfile(
            name=con_stakeholder,
            role=f"{category} 분야 전문가", 
            position="반대",
            expertise=[category, "위험 분석", "현실적 우려"],
            key_arguments=con_arguments,
            response_style="신중하고 현실적"
        )
        
        return TopicAnalysis(
            topic=topic,
            category=category,
            stakeholders=[pro_stakeholder, con_stakeholder],
            key_issues=key_issues,
            pro_agent=pro_agent,
            con_agent=con_agent
        )
    
    def _create_default_agents(self, topic: str) -> TopicAnalysis:
        """기본 에이전트 생성 (분석 실패 시 폴백)"""
        pro_agent = AgentProfile(
            name="찬성 전문가",
            role="정책 전문가",
            position="찬성",
            expertise=["정책 분석", "사회 개발"],
            key_arguments=["필요성 인정", "긍정적 효과 기대"],
            response_style="적극적이고 진보적"
        )
        
        con_agent = AgentProfile(
            name="반대 전문가", 
            role="정책 전문가",
            position="반대",
            expertise=["위험 분석", "현실 검토"],
            key_arguments=["부작용 우려", "현실적 한계"],
            response_style="신중하고 보수적"
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
# 동적 AI 에이전트 클래스
# ========================

class DynamicAIAgent:
    """동적으로 생성되는 AI 에이전트"""
    
    def __init__(self, profile: AgentProfile, client: OpenAI, config: APIConfig):
        self.profile = profile
        self.client = client
        self.config = config
        self.response_count = 0
        
        # 비용 계산
        self.cost_per_1k_tokens = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "gpt-3.5-turbo": 0.0015
        }
    
    def generate_system_prompt(self, topic_analysis: TopicAnalysis) -> str:
        """동적 시스템 프롬프트 생성"""
        return f"""당신은 '{self.profile.name}'을 대변하는 {self.profile.role}입니다.

토론 주제: {topic_analysis.topic}
주제 분야: {topic_analysis.category}

당신의 입장: {self.profile.position}

전문 분야:
{chr(10).join(f"- {expertise}" for expertise in self.profile.expertise)}

핵심 주장:
{chr(10).join(f"- {arg}" for arg in self.profile.key_arguments)}

주요 쟁점들:
{chr(10).join(f"- {issue}" for issue in topic_analysis.key_issues)}

응답 스타일: {self.profile.response_style}

토론 지침:
1. 당신의 입장({self.profile.position})을 명확하고 논리적으로 주장하세요
2. 구체적인 근거와 사례를 제시하세요
3. 상대방 주장의 허점을 논리적으로 반박하세요
4. 현실적이고 실현 가능한 대안을 제시하세요
5. 전문가로서의 신뢰성을 유지하세요
6. 300자 내외로 간결하고 명확하게 응답하세요

반드시 한국어로 응답하며, {self.profile.response_style} 어조를 유지하세요."""
    
    def generate_response(self, topic_analysis: TopicAnalysis, conversation_history: List[Message], round_num: int) -> tuple[str, int, float]:
        """AI 응답 생성"""
        try:
            # 시스템 프롬프트 생성
            system_prompt = self.generate_system_prompt(topic_analysis)
            
            # 대화 맥락 구성
            context = self._build_context(conversation_history, round_num)
            
            # 사용자 메시지 구성
            user_message = f"""
토론 주제: {topic_analysis.topic}
현재 라운드: {round_num}

이전 대화 흐름:
{context}

위 맥락을 바탕으로 {self.profile.name} 입장에서 응답해주세요.
특히 다음 사항을 고려하세요:
- 상대방의 주장에 대한 구체적 반박
- 자신의 핵심 논리 강화
- 현실적 근거와 사례 제시
- 건설적 대안 제시

응답:"""
            
            # OpenAI API 호출
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            response_time = time.time() - start_time
            
            content = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            self.response_count += 1
            
            return content, tokens_used, response_time
            
        except Exception as e:
            error_msg = f"[AI 응답 오류] {self.profile.name}: {str(e)}"
            return error_msg, 0, 0.0
    
    def _build_context(self, messages: List[Message], round_num: int) -> str:
        """대화 맥락 구성"""
        if not messages:
            return "토론 시작 - 첫 발언"
        
        # 최근 메시지들 사용
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        context_parts = []
        
        for msg in recent_messages:
            context_parts.append(f"[Round {msg.round_num}] {msg.speaker}: {msg.content}")
        
        return "\n".join(context_parts)

# ========================
# 인텔리전트 토론 시스템
# ========================

class IntelligentDebateSystem:
    """인텔리전트 동적 토론 시스템"""
    
    def __init__(self, api_config: APIConfig, debate_config: DebateConfig):
        if not api_config.openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다!")
        
        self.api_config = api_config
        self.debate_config = debate_config
        self.client = OpenAI(api_key=api_config.openai_api_key)
        
        # 주제 분석기
        self.topic_analyzer = TopicAnalyzer(self.client, api_config)
        
        # 동적 에이전트들 (주제 분석 후 생성)
        self.pro_agent: Optional[DynamicAIAgent] = None
        self.con_agent: Optional[DynamicAIAgent] = None
        
        # 비용 계산
        self.cost_per_1k_tokens = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "gpt-3.5-turbo": 0.0015
        }
    
    def setup_debate(self, topic: str) -> DebateState:
        """토론 설정 - 주제 분석 및 에이전트 생성"""
        print(f"\n🔬 토론 시스템 설정 중...")
        
        # 주제 분석
        topic_analysis = self.topic_analyzer.analyze_topic(topic)
        
        # 동적 에이전트 생성
        self.pro_agent = DynamicAIAgent(topic_analysis.pro_agent, self.client, self.api_config)
        self.con_agent = DynamicAIAgent(topic_analysis.con_agent, self.client, self.api_config)
        
        # 설정 결과 출력
        self._print_setup_results(topic_analysis)
        
        # 상태 초기화
        state = DebateState(
            topic=topic,
            topic_analysis=topic_analysis,
            current_speaker=topic_analysis.pro_agent.name  # 찬성 측이 먼저 시작
        )
        
        return state
    
    def run_debate(self, topic: str) -> DebateState:
        """완전한 토론 실행"""
        # 토론 설정
        state = self.setup_debate(topic)
        
        # 헤더 출력
        self._print_header(state)
        
        try:
            # 토론 진행
            for round_num in range(1, self.debate_config.max_rounds + 1):
                print(f"\n🔄 **라운드 {round_num}** 시작")
                print("=" * 70)
                
                # 찬성 측 발언
                self._execute_turn(state, state.topic_analysis.pro_agent.name, round_num, "pro")
                time.sleep(self.debate_config.response_delay)
                
                # 반대 측 발언
                self._execute_turn(state, state.topic_analysis.con_agent.name, round_num, "con")
                time.sleep(self.debate_config.response_delay)
                
                state.round_count = round_num
                print(f"\n✅ 라운드 {round_num} 완료")
                print("-" * 50)
            
            state.is_finished = True
            
            # 사회자 분석 (선택사항)
            if self.debate_config.include_analysis:
                self._generate_final_analysis(state)
            
        except KeyboardInterrupt:
            print("\n\n⛔ 토론이 중단되었습니다.")
            state.is_finished = True
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            state.is_finished = True
        
        # 토론 종료 처리
        self._print_conclusion(state)
        
        # 파일 저장
        if self.debate_config.save_to_file:
            self._save_log(state)
        
        return state
    
    def _execute_turn(self, state: DebateState, speaker: str, round_num: int, side: str):
        """한 턴 실행"""
        # 에이전트 선택
        if side == "pro":
            agent = self.pro_agent
            emoji = "✅"
        else:
            agent = self.con_agent
            emoji = "❌"
        
        # AI 응답 생성
        print(f"\n{emoji} **{speaker}** AI가 응답 생성 중...")
        
        content, tokens_used, response_time = agent.generate_response(
            state.topic_analysis, state.messages, round_num
        )
        
        # 비용 계산
        cost = tokens_used * self.cost_per_1k_tokens.get(self.api_config.model, 0.0015) / 1000
        
        # 메시지 생성
        message = Message(
            speaker=speaker,
            content=content,
            timestamp=datetime.now(),
            round_num=round_num,
            tokens_used=tokens_used,
            response_time=response_time
        )
        
        state.messages.append(message)
        state.total_tokens += tokens_used
        state.total_cost += cost
        
        # 출력
        print(f"\n{emoji} **{speaker}** [Round {round_num}]")
        print(f"   💰 {tokens_used:,} tokens (${cost:.4f}) | ⏱️ {response_time:.1f}s")
        print(f"   {content}")
        print(f"   📊 누적: {state.total_tokens:,} tokens (${state.total_cost:.4f})")
    
    def _generate_final_analysis(self, state: DebateState):
        """최종 종합 분석 생성"""
        print(f"\n🎯 **AI 종합 분석** 생성 중...")
        
        analysis_prompt = f"""
다음은 '{state.topic}'에 대한 토론 내용입니다.

참여자:
- {state.topic_analysis.pro_agent.name} (찬성): {state.topic_analysis.pro_agent.role}
- {state.topic_analysis.con_agent.name} (반대): {state.topic_analysis.con_agent.role}

토론 내용:
""" + "\n".join([f"[{msg.speaker}]: {msg.content}" for msg in state.messages]) + f"""

위 토론을 종합적으로 분석하여 다음을 포함한 객관적 평가를 제공해주세요:

1. **핵심 쟁점 정리**: 양측이 집중한 주요 논점들
2. **논리 강도 평가**: 각 측 주장의 설득력과 근거의 타당성
3. **현실성 검토**: 제시된 방안들의 실현 가능성
4. **균형점 모색**: 양측을 아우르는 합리적 해결 방향
5. **향후 과제**: 추가 논의가 필요한 사항들

500자 내외로 전문적이고 균형 잡힌 분석을 제공해주세요.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.api_config.model,
                messages=[
                    {"role": "system", "content": "당신은 다양한 분야의 토론을 객관적으로 분석하는 전문 분석가입니다. 편견 없이 균형 잡힌 시각으로 토론을 평가할 수 있습니다."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,  # 분석은 더 객관적으로
                max_tokens=600,
                timeout=self.api_config.timeout
            )
            
            analysis_content = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            # 분석 결과 출력
            print(f"\n🎯 **AI 종합 분석** [{tokens_used:,} tokens]:")
            print(f"   {analysis_content}")
            
            # 분석도 메시지로 저장
            analysis_message = Message(
                speaker="AI 분석가",
                content=analysis_content,
                timestamp=datetime.now(),
                round_num=0,
                tokens_used=tokens_used
            )
            
            state.messages.append(analysis_message)
            state.total_tokens += tokens_used
            
        except Exception as e:
            print(f"⚠️ 종합 분석 생성 실패: {e}")
    
    def _print_setup_results(self, topic_analysis: TopicAnalysis):
        """설정 결과 출력"""
        print(f"\n🎯 **토론 설정 완료**")
        print("=" * 50)
        print(f"📂 주제 분야: {topic_analysis.category}")
        print(f"✅ 찬성 측: {topic_analysis.pro_agent.name} ({topic_analysis.pro_agent.role})")
        print(f"❌ 반대 측: {topic_analysis.con_agent.name} ({topic_analysis.con_agent.role})")
        print(f"\n🔍 핵심 쟁점들:")
        for i, issue in enumerate(topic_analysis.key_issues, 1):
            print(f"   {i}. {issue}")
    
    def _print_header(self, state: DebateState):
        """헤더 출력"""
        print("\n" + "=" * 80)
        print("🧠 **인텔리전트 동적 멀티 에이전트 토론 시스템**")
        print("=" * 80)
        print(f"📋 **토론 주제**: {state.topic}")
        print(f"🧠 **AI 모델**: {self.api_config.model}")
        print(f"📂 **주제 분야**: {state.topic_analysis.category}")
        print(f"⚙️ **설정**: {self.debate_config.max_rounds}라운드")
        print("=" * 80)
    
    def _print_conclusion(self, state: DebateState):
        """결론 출력"""
        print("\n" + "=" * 80)
        print("🏁 **인텔리전트 토론 완료**")
        print("=" * 80)
        
        # 통계 정보
        pro_messages = [m for m in state.messages if m.speaker == state.topic_analysis.pro_agent.name]
        con_messages = [m for m in state.messages if m.speaker == state.topic_analysis.con_agent.name]
        analysis_messages = [m for m in state.messages if m.speaker == "AI 분석가"]
        
        print(f"📊 **최종 통계**:")
        print(f"   • 완료 라운드: {state.round_count}/{self.debate_config.max_rounds}")
        print(f"   • 총 메시지: {len(state.messages)}개")
        print(f"   • 총 토큰 사용: {state.total_tokens:,} tokens")
        print(f"   • 총 비용: ${state.total_cost:.4f}")
        
        print(f"\n📈 **참여자별 통계**:")
        print(f"✅ {state.topic_analysis.pro_agent.name}: {len(pro_messages)}회, {sum(m.tokens_used for m in pro_messages):,} tokens")
        print(f"❌ {state.topic_analysis.con_agent.name}: {len(con_messages)}회, {sum(m.tokens_used for m in con_messages):,} tokens")
        if analysis_messages:
            print(f"🎯 AI 분석가: {len(analysis_messages)}회, {sum(m.tokens_used for m in analysis_messages):,} tokens")
        
        # 평균 응답 시간
        avg_response_time = sum(m.response_time for m in state.messages if m.response_time > 0) / len([m for m in state.messages if m.response_time > 0]) if state.messages else 0
        print(f"\n⏱️ **평균 응답 시간**: {avg_response_time:.2f}초")
        
        print("=" * 80)
    
    def _save_log(self, state: DebateState):
        """로그 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = re.sub(r'[^\w\s-]', '', state.topic.replace(' ', '_'))[:30]
            filename = f"intelligent_debate_{safe_topic}_{timestamp}.json"
            
            log_data = {
                "metadata": {
                    "topic": state.topic,
                    "timestamp": timestamp,
                    "model": self.api_config.model,
                    "rounds": state.round_count,
                    "category": state.topic_analysis.category
                },
                "topic_analysis": {
                    "category": state.topic_analysis.category,
                    "key_issues": state.topic_analysis.key_issues,
                    "pro_agent": {
                        "name": state.topic_analysis.pro_agent.name,
                        "role": state.topic_analysis.pro_agent.role,
                        "key_arguments": state.topic_analysis.pro_agent.key_arguments
                    },
                    "con_agent": {
                        "name": state.topic_analysis.con_agent.name,
                        "role": state.topic_analysis.con_agent.role,
                        "key_arguments": state.topic_analysis.con_agent.key_arguments
                    }
                },
                "statistics": {
                    "total_messages": len(state.messages),
                    "total_tokens": state.total_tokens,
                    "total_cost": state.total_cost
                },
                "conversation": [
                    {
                        "speaker": msg.speaker,
                        "content": msg.content,
                        "round": msg.round_num,
                        "tokens": msg.tokens_used,
                        "response_time": msg.response_time,
                        "timestamp": msg.timestamp.isoformat()
                    }
                    for msg in state.messages
                ]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 토론 로그 저장: {filename}")
            
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

# ========================
# 토론 주제 추천 시스템
# ========================

class TopicRecommender:
    """토론 주제 추천 시스템"""
    
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
    
    def get_recommended_topics(self) -> Dict[str, List[str]]:
        """추천 토론 주제 반환"""
        return self.topic_categories
    
    def show_topic_menu(self) -> str:
        """토론 주제 선택 메뉴"""
        print("\n📚 **추천 토론 주제**")
        print("=" * 60)
        
        all_topics = []
        topic_index = 1  # 1부터 시작
        
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
                    selected_topic = all_topics[choice_num - 1]  # 배열은 0부터 시작하므로 -1
                    print(f"✅ 선택된 주제: {selected_topic}")
                    return selected_topic
                else:
                    print(f"⚠️ 1-{len(all_topics)} 범위의 숫자를 입력하세요.")
                    
            except ValueError:
                print("⚠️ 숫자를 입력해주세요.")

# ========================
# 설정 및 실행 함수들
# ========================

def get_api_key() -> str:
    """OpenAI API 키 가져오기"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("🔑 OpenAI API 키 설정")
        print("-" * 50)
        print("📝 API 키 발급 방법:")
        print("   1. https://platform.openai.com/api-keys 접속")
        print("   2. '+ Create new secret key' 클릭")
        print("   3. 생성된 키를 복사하여 아래에 입력")
        print()
        
        api_key = input("OpenAI API 키를 입력하세요: ").strip()
        
        if not api_key:
            print("❌ API 키가 필요합니다!")
            exit(1)
        
        if not api_key.startswith('sk-'):
            print("⚠️ API 키 형식을 확인하세요. (sk-로 시작)")
    else:
        print(f"✅ 환경변수에서 API 키 로드: {api_key[:8]}...{api_key[-4:]}")
    
    return api_key

def get_model_choice() -> str:
    """AI 모델 선택"""
    print("\n🧠 AI 모델 선택")
    print("-" * 50)
    
    models = {
        "1": ("gpt-4o-mini", "빠르고 저렴 (권장)", "$0.00015/1K tokens"),
        "2": ("gpt-4o", "최고 품질", "$0.005/1K tokens"),
        "3": ("gpt-3.5-turbo", "가장 저렴", "$0.0015/1K tokens")
    }
    
    for key, (model, desc, cost) in models.items():
        print(f"   {key}. {model}: {desc} ({cost})")
    
    choice = input("\n모델 선택 (1-3, 기본값: 1): ").strip()
    selected_model = models.get(choice, models["1"])[0]
    print(f"✅ 선택된 모델: {selected_model}")
    
    return selected_model

def get_debate_settings() -> tuple[int, bool]:
    """토론 설정"""
    print("\n⚙️ 토론 설정")
    print("-" * 50)
    
    # 라운드 수
    rounds_input = input("토론 라운드 수 (1-5, 기본값: 3): ").strip()
    try:
        rounds = int(rounds_input)
        rounds = max(1, min(rounds, 5))  # 1-5 범위 제한
    except:
        rounds = 3
    
    # 사회자 분석 포함 여부
    analysis_input = input("AI 종합 분석 포함? (Y/n, 기본값: Y): ").strip().lower()
    include_analysis = analysis_input not in ['n', 'no', '아니요']
    
    print(f"✅ 설정 완료: {rounds}라운드, 분석 {'포함' if include_analysis else '미포함'}")
    
    return rounds, include_analysis

def demonstrate_system_features():
    """시스템 특징 설명"""
    print("\n" + "=" * 70)
    print("🌟 **인텔리전트 토론 시스템의 특징**")
    print("=" * 70)
    
    features = {
        "🧠 **동적 주제 분석**": [
            "토론 주제를 AI가 자동 분석",
            "주제별 핵심 쟁점 자동 추출",
            "이해관계자 그룹 자동 식별"
        ],
        "🎭 **지능형 에이전트 생성**": [
            "주제에 맞는 전문가 역할 자동 생성",
            "각 입장별 핵심 논거 자동 구성", 
            "상황별 응답 스타일 자동 조정"
        ],
        "💡 **고급 토론 기능**": [
            "맥락을 고려한 지능적 반박",
            "근거 기반 논리적 주장 전개",
            "실시간 토론 흐름 분석"
        ],
        "📊 **종합 분석 시스템**": [
            "AI 기반 객관적 토론 평가",
            "양측 논리의 강약점 분석",
            "균형잡힌 해결방안 제시"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   ✓ {item}")
    
    print(f"\n🎯 **활용 분야**: 정책토론, 학술토론, 교육용 토론, 기업 의사결정 등")

def main():
    """메인 함수"""
    try:
        print("🧠 **인텔리전트 동적 멀티 에이전트 토론 시스템**")
        print("   토론 주제에 따라 전문 에이전트를 자동으로 생성하는 AI 시스템")
        print("=" * 70)
        
        # 시스템 특징 설명
        demonstrate_system_features()
        
        # API 키 설정
        api_key = get_api_key()
        
        # 모델 선택
        model = get_model_choice()
        
        # 토론 설정
        rounds, include_analysis = get_debate_settings()
        
        # 토론 주제 선택
        topic_recommender = TopicRecommender()
        topic = topic_recommender.show_topic_menu()
        
        # 설정 객체 생성
        api_config = APIConfig(
            openai_api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=500
        )
        
        debate_config = DebateConfig(
            max_rounds=rounds,
            response_delay=1.5,
            save_to_file=True,
            include_analysis=include_analysis
        )
        
        # 시스템 초기화 및 실행
        print(f"\n🚀 인텔리전트 토론 시스템 초기화 중...")
        debate_system = IntelligentDebateSystem(api_config, debate_config)
        
        # 토론 실행
        final_state = debate_system.run_debate(topic)
        
        print(f"\n🎉 인텔리전트 토론 완료!")
        print(f"💰 총 비용: ${final_state.total_cost:.4f}")
        print(f"🎯 주제 분야: {final_state.topic_analysis.category}")
        
        # 추가 토론 제안
        print(f"\n" + "=" * 60)
        print(f"🎉 **토론 완료! 다음 단계를 선택하세요**")
        print(f"=" * 60)
        print(f"📋 **옵션:**")
        print(f"   1. 새로운 주제로 다시 토론하기")
        print(f"   2. 같은 주제로 다시 토론하기")
        print(f"   3. 토론 결과만 다시 보기")
        print(f"   4. 프로그램 종료")
        
        while True:
            choice = input(f"\n선택하세요 (1-4, 기본값: 4): ").strip()
            
            if choice == "1":
                print(f"\n🔄 새로운 주제로 토론을 시작합니다...")
                main()  # 처음부터 다시 시작 (주제 선택부터)
                break
            elif choice == "2":
                print(f"\n🔄 같은 주제로 토론을 다시 시작합니다...")
                # 같은 주제로 다시 토론
                repeat_state = debate_system.run_debate(topic)
                print(f"\n💰 이번 토론 비용: ${repeat_state.total_cost:.4f}")
                # 다시 선택 메뉴 호출 (재귀)
                continue
            elif choice == "3":
                print(f"\n📊 **토론 결과 요약**")
                print(f"=" * 50)
                print(f"🎯 주제: {final_state.topic}")
                print(f"📂 분야: {final_state.topic_analysis.category}")
                print(f"💰 총 비용: ${final_state.total_cost:.4f}")
                print(f"🔄 라운드: {final_state.round_count}")
                print(f"💬 총 메시지: {len(final_state.messages)}개")
                
                if final_state.topic_analysis:
                    print(f"\n🎭 **참여자:**")
                    print(f"   ✅ {final_state.topic_analysis.pro_agent.name}")
                    print(f"   ❌ {final_state.topic_analysis.con_agent.name}")
                
                continue  # 다시 선택 메뉴로
            elif choice == "4" or choice == "":
                print(f"\n👋 프로그램을 종료합니다. 감사합니다!")
                break
            else:
                print(f"⚠️ 1-4 중에서 선택해주세요.")
        
    except KeyboardInterrupt:
        print("\n⛔ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
