"""
간단한 멀티 에이전트 협업 토론 시스템
Python 3.13 호환성 최적화 버전
"""

import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any

# ========================
# 데이터 구조 정의
# ========================

@dataclass
class Message:
    speaker: str
    content: str
    timestamp: float

@dataclass 
class DebateState:
    messages: List[Message]
    current_speaker: str
    round_count: int
    topic: str
    is_finished: bool = False

# ========================
# 에이전트 클래스들
# ========================

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
    
    def generate_response(self, topic: str, conversation_history: List[Message]) -> str:
        """기본 응답 생성 메서드"""
        return f"{self.name}의 의견입니다."

class DoctorUnionAgent(BaseAgent):
    def __init__(self):
        super().__init__("의사협회", "의대 정원 확대 반대")
        self.arguments = [
            "의대 정원 확대보다는 현재 의사들의 지역 배치 문제 해결이 우선입니다. 무분별한 확대는 의료 교육의 질 저하를 초래할 수 있습니다.",
            "OECD 통계를 단순 비교하는 것은 부적절합니다. 의료 접근성과 분배 문제가 더 중요한 이슈입니다.",
            "급작스러운 정원 확대는 의료진 양성 시스템에 과부하를 줄 것입니다. 근무 환경 개선과 인센티브 정책이 먼저 필요합니다.",
            "지방 의료 문제는 단순히 의사 수를 늘린다고 해결되지 않습니다. 구조적 접근이 필요합니다."
        ]
    
    def generate_response(self, topic: str, conversation_history: List[Message]) -> str:
        # 이전 정부 발언에 대한 반박 요소 추가
        gov_mentions = [msg for msg in conversation_history if msg.speaker == "정부"]
        
        base_response = random.choice(self.arguments)
        
        if gov_mentions:
            counter_points = [
                "정부의 주장과 달리,",
                "정부가 제시한 통계에 대해 다른 관점에서 보면,",
                "정부 정책의 한계는"
            ]
            base_response = f"{random.choice(counter_points)} {base_response}"
        
        return base_response

class GovernmentAgent(BaseAgent):
    def __init__(self):
        super().__init__("정부", "의대 정원 확대 찬성")
        self.arguments = [
            "인구 고령화로 의료 수요가 급증하고 있습니다. OECD 평균 대비 우리나라 의사 수는 현저히 부족한 상황입니다.",
            "필수의료와 지방의료 인력 부족이 심각합니다. 체계적인 정원 확대를 통해 이를 해결하겠습니다.",
            "의료진 근로시간 단축과 의료 수요 증가를 고려할 때, 의사 수 확충은 불가피합니다.",
            "지방의대 우선 증원과 장학금 확대 등을 통해 지역 의료 인력을 체계적으로 양성하겠습니다."
        ]
    
    def generate_response(self, topic: str, conversation_history: List[Message]) -> str:
        # 이전 의사협회 발언에 대한 대응 요소 추가
        doctor_mentions = [msg for msg in conversation_history if msg.speaker == "의사협회"]
        
        base_response = random.choice(self.arguments)
        
        if doctor_mentions:
            supporting_data = [
                "객관적 데이터를 보면,",
                "국제 비교 연구에 따르면,",
                "정부의 정책 연구 결과,"
            ]
            base_response = f"{random.choice(supporting_data)} {base_response}"
        
        return base_response

class ModeratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("사회자", "토론 진행")
    
    def should_continue(self, state: DebateState) -> bool:
        """토론 계속 여부 판단"""
        return state.round_count < 4 and len(state.messages) < 12
    
    def get_next_speaker(self, current_speaker: str) -> str:
        """다음 발언자 결정"""
        if current_speaker == "정부":
            return "의사협회"
        else:
            return "정부"
    
    def generate_summary(self, conversation_history: List[Message]) -> str:
        """토론 요약 생성"""
        gov_count = len([msg for msg in conversation_history if msg.speaker == "정부"])
        doctor_count = len([msg for msg in conversation_history if msg.speaker == "의사협회"])
        
        return f"""
토론 요약:
- 총 {len(conversation_history)}개의 발언
- 정부 측 발언: {gov_count}회
- 의사협회 측 발언: {doctor_count}회

주요 쟁점:
• 정부: 의사 수 부족, OECD 통계, 고령화 대응
• 의사협회: 교육 질 저하 우려, 지역 배치 문제, 구조적 해결 필요성

양측 모두 국민 건강 향상이라는 공통 목표를 가지고 있으나, 
접근 방법에서 차이를 보이고 있습니다.
"""

# ========================
# 토론 시스템 클래스
# ========================

class DebateSystem:
    def __init__(self):
        self.doctor_agent = DoctorUnionAgent()
        self.government_agent = GovernmentAgent()
        self.moderator = ModeratorAgent()
        
    def run_debate(self, topic: str) -> DebateState:
        """토론 실행"""
        print("=" * 70)
        print("🎭 **간단한 멀티 에이전트 협업 토론 시스템**")
        print("=" * 70)
        print(f"📋 **토론 주제**: {topic}")
        print("🤖 **시스템**: 안정성 최적화 버전")
        print("=" * 70)
        
        # 초기 상태 설정
        state = DebateState(
            messages=[],
            current_speaker="정부",  # 정부가 먼저 시작
            round_count=0,
            topic=topic
        )
        
        # 토론 진행
        while not state.is_finished:
            try:
                # 현재 발언자에 따른 응답 생성
                if state.current_speaker == "정부":
                    response = self.government_agent.generate_response(topic, state.messages)
                    speaker_emoji = "🏛️"
                else:
                    response = self.doctor_agent.generate_response(topic, state.messages)
                    speaker_emoji = "🏥"
                
                # 메시지 추가
                message = Message(
                    speaker=state.current_speaker,
                    content=response,
                    timestamp=time.time()
                )
                state.messages.append(message)
                
                # 발언 출력
                print(f"\n{speaker_emoji} **{state.current_speaker}**:")
                print(f"   {response}")
                print("-" * 50)
                
                # 잠깐 대기 (읽기 편하게)
                time.sleep(0.5)
                
                # 다음 발언자 결정
                state.current_speaker = self.moderator.get_next_speaker(state.current_speaker)
                
                # 라운드 카운트 업데이트 (한 쌍의 발언이 끝날 때마다)
                if state.current_speaker == "정부":
                    state.round_count += 1
                
                # 토론 종료 조건 확인
                if not self.moderator.should_continue(state):
                    state.is_finished = True
                    
            except KeyboardInterrupt:
                print("\n\n⛔ 사용자에 의해 토론이 중단되었습니다.")
                state.is_finished = True
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                state.is_finished = True
        
        # 토론 종료 및 요약
        print("\n" + "=" * 70)
        print("🏁 **토론 종료**")
        print("=" * 70)
        
        # 사회자 요약
        summary = self.moderator.generate_summary(state.messages)
        print(f"\n⚖️ **사회자 요약**:")
        print(summary)
        
        return state

# ========================
# 확장 기능 및 유틸리티
# ========================

def demonstrate_features():
    """시스템 특징 및 확장성 설명"""
    print("\n" + "=" * 70)
    print("🔧 **시스템 특징 및 확장 가능성**")
    print("=" * 70)
    
    features = {
        "✅ **현재 구현된 기능**": [
            "멀티 에이전트 상태 관리",
            "순차적 대화 플로우 제어", 
            "동적 응답 생성 (컨텍스트 고려)",
            "자동 토론 종료 및 요약",
            "오류 처리 및 안정성 보장"
        ],
        "🚀 **LangGraph 연동 시 추가 가능**": [
            "복잡한 조건부 라우팅",
            "병렬 에이전트 실행",
            "외부 도구 통합 (검색, DB 등)",
            "상태 저장 및 복원",
            "실시간 모니터링"
        ],
        "🌐 **웹 인터페이스 확장**": [
            "Streamlit 대시보드",
            "FastAPI REST API",
            "실시간 채팅 인터페이스",
            "토론 기록 시각화",
            "사용자 참여 기능"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

def create_custom_debate(topic: str, agents_config: Dict[str, Any]):
    """커스텀 토론 생성 예시"""
    print(f"\n🎯 **커스텀 토론 예시**: {topic}")
    print("💡 이와 같은 방식으로 다양한 주제와 에이전트 구성이 가능합니다.")

# ========================
# 메인 실행부
# ========================

def main():
    """메인 실행 함수"""
    try:
        # 토론 시스템 초기화
        debate_system = DebateSystem()
        
        # 기본 토론 실행
        topic = "2024년 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?"
        final_state = debate_system.run_debate(topic)
        
        # 시스템 특징 설명
        demonstrate_features()
        
        # 커스텀 토론 예시
        custom_examples = [
            "환경세 도입의 필요성",
            "AI 규제 정책의 적정선",
            "원격근무 제도의 의무화"
        ]
        
        print(f"\n📚 **다른 토론 주제 예시**:")
        for i, example in enumerate(custom_examples, 1):
            print(f"  {i}. {example}")
        
        print(f"\n🎉 **총 {len(final_state.messages)}개의 발언으로 토론이 완료되었습니다!**")
        
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")
    finally:
        print("\n🔚 프로그램을 안전하게 종료합니다.")

if __name__ == "__main__":
    main()
