import rclpy
from pinky_interfaces.srv import Agent
from rclpy.node import Node
import readline

class AgentClient(Node):
    def __init__(self):
        super().__init__('agent_client')
        self.cli = self.create_client(Agent, 'llm_agent')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the service...')

        self.req = Agent.Request()

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
