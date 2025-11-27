import rclpy
from rclpy.node import Node
from pinky_llm.robot_tools import ToolSet, create_tools
from pinky_interfaces.srv import Agent
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm_dir = get_package_share_directory('pinky_llm')
nav2_dir = get_package_share_directory('pinky_navigation')
env_file_path = Path(llm_dir) / '.env'
load_dotenv(dotenv_path=env_file_path)

prompt_file = Path(llm_dir) / 'params/prompt.yaml'
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompt_data = yaml.safe_load(f)    

class AgentLLM(Node):
    def __init__(self):
        super().__init__('agent_llm')

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_data["system"]),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        self.srv = self.create_service(Agent, 'llm_agent', self.handle_question)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
        yaml_file = Path(nav2_dir) / 'params/points.yaml'
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        places = {name: (info["x"], info["y"], info["qz"], info["qw"]) for name, info in config["places"].items()}
        tool_set = ToolSet(places)
        tool_list = create_tools(tool_set)

        self.agent = create_tool_calling_agent(self.llm, tool_list, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tool_list, verbose=True)

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
            # response_match = re.search(r"ANSWER:\s*([\s\S]*)", answer, re.IGNORECASE)
            # response.answer = response_match.group(1).strip() if response_match else "[ERR] No answer parsed"
            response.answer = answer
        except Exception as e:
            self.get_logger().info(e)
            response.answer = "잘 이해하지 못했어요.. 자세하게 물어봐 주시겠어요?"
        return response
    
def main(args=None):
    rclpy.init(args=args)
    agent = AgentLLM()
    try:
        rclpy.spin(agent) 
    finally:
        agent.destroy_node()
        rclpy.shutdown()
