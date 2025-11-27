import rclpy
from pinky_interfaces.srv import AskAgent
from rclpy.node import Node
import readline

class AgentClient(Node):
    def __init__(self):
        super().__init__('agent_client')
        self.cli = self.create_client(AskAgent, 'ask_agent')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('서비스 대기 중...')
        self.req = AskAgent.Request()

    def ask(self, question: str) -> str:
        self.req.question = question
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().answer

def main():
    rclpy.init()
    client = AgentClient()

    try:
        while True:
            q = input("💬 질문: ")
            answer = client.ask(q)
            print("🤖:", answer, "\n")
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
