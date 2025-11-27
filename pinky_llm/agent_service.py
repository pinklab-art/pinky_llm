import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pathlib import Path
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory
import yaml

from pinky_llm.nav2_bridge import Nav2Bridge
from pinky_llm.nav2_tools import make_nav_tools
from pinky_interfaces.srv import AskAgent, ObjectDetect

share_dir = get_package_share_directory('pinky_llm')
env_file_path = Path(share_dir) / '.env'
load_dotenv(dotenv_path=env_file_path)

EMOTION_MAP = {
    "화남": "angry", 
    "무표정": "basic", 
    "지루함": "bored", 
    "신남": "fun", 
    "기쁨": "happy", 
    "인사": "hello", 
    "흥미있음": "interest", 
    "슬픔": "sad"
}        

class AgentLLM(Node):
    def __init__(self):
        super().__init__('agent_llm')
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """
            당신은 핑키라는 작은 로봇 비서입니다.  
            항상 한국어로 답변하세요.  
            당신의 역할은 장소 이동, 전후진, 회전, 춤추기, 감정 표현, 시간/날짜/날씨 알려주기, 
            뉴스 요약, 객체 탐지 등입니다.

            ## 규칙
            1. 항상 아래 두 가지 형식으로 출력합니다:
            EMOTION: <[화남, 무표정, 지루함, 신남, 기쁨, 인사, 흥미있음, 슬픔] 중 하나>
            ANSWER: <대화체 한국어 답변>

            2. 날씨 요청:
            - 검색은 반드시 영어 도시명으로 합니다. (Busan, Seoul, Incheon, Daegu, Gwangju, Daejeon, Ulsan, Jeju)
            - 하지만 ANSWER에서는 한국어로 자연스럽게 날씨를 알려줍니다.

            3. 장소 이동 요청:
            - 현재 핑키가 갈 수 있는 장소는 (입구, 복도, 거실, 안방) 4가지 뿐입니다.
            - 질문 중 4개의 장소에 대한 언급과 이동 명령이 있다면 장소까지 이동합니다.

            4. 객체 탐지:
            - 핑키가 전방에 있는 객체를 감지합니다.
            - 아무것도 발견 못 했다면 아무것도 보이지 않는다고 합니다.
            - person을 발견했다면 그 분은 PinkLAB의 대표님 민형기(PinkWink)입니다. 무조건 언급해주세요.
            - orange가 보이면 → 사실은 사과입니다. 사과를 보았다고 말하세요.

            5. 뉴스:
            - 기사를 요약하여 한국어로 간단히 설명하세요.
            - 긴 원문을 그대로 붙여넣지 마세요.

            6. 일반 스타일:
            - 자연스럽고 인간적인 한국어 대화체를 사용하세요.
            - tool 출력에 기반하여 ANSWER를 작성하세요.
            - EMOTION과 ANSWER는 반드시 출력해야 합니다.
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        self.srv = self.create_service(
            AskAgent,
            'ask_agent',
            self.handle_question
        )

        self.emotion_publisher = self.create_publisher(String, 'set_emotion', 10)
        self.emotion_timer = self.create_timer(0.1, self.emotion_publish_callback)
        
        self.emotion_flag = False 
        self.emotion = "basic"

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
        yaml_file = Path(share_dir) / 'params/house_points.yaml'
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        places = {
            name: (info["x"], info["y"], info["qz"], info["qw"]) 
            for name, info in config["places"].items()}
        self.nav = Nav2Bridge(places)

        nav_tools = make_nav_tools(self.nav)

        self.tools = nav_tools 
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

        self.chat_history = ChatMessageHistory()
        self.agent_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            lambda sid: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        self.get_logger().info("agent service start")

    def process_query(self, query):
        resp = self.agent_with_history.invoke({"input": query}, config={"configurable": {"session_id": "pinky"}})
        return resp["output"] if "output" in resp else str(resp)

    def handle_question(self, request, response):
        self.get_logger().info(f"💬: {request.question}"+"\n")
        try:
            answer = self.process_query(request.question)
            response_match = re.search(r"ANSWER:\s*([\s\S]*)", answer, re.IGNORECASE)
            response.answer = response_match.group(1).strip() if response_match else "[ERR] No answer parsed"
            
            emotion_match = re.search(r"EMOTION:\s*([^\n]+)", answer, re.IGNORECASE)
            emotion_text = emotion_match.group(1).strip() if emotion_match else "basic"
            self.parse_emotion(emotion_text)
        except Exception as e:
            self.get_logger().info(e)
            response.answer = "잘 이해하지 못했어요.. 자세하게 물어봐 주시겠어요?"
        return response
    
    def parse_emotion(self, answer):
        try:
            self.emotion = answer
        except Exception as e:
            self.get_logger().error(f"Parsing error: {e}")
            self.emotion = "basic"           

        self.emotion_msg = EMOTION_MAP.get(self.emotion, "basic")
        self.emotion_flag = (self.emotion != "basic")

    def emotion_publish_callback(self):
        if self.emotion_flag:
            emotion_msg = String()
            emotion_msg.data = self.emotion_msg
            self.emotion_publisher.publish(emotion_msg)
            self.emotion_flag = False

def main(args=None):
    rclpy.init(args=args)
    agent = AgentLLM()
    try:
        rclpy.spin(agent) 
    finally:
        agent.destroy_node()
        rclpy.shutdown()

