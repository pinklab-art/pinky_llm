import math
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from langchain.tools import StructuredTool
from typing import List


def create_tools(tool_set) -> List[StructuredTool]:
    tool_specs = [
        {
            "name": "get_current_location",
            "description": "사용자가 현재 로봇의 위치나 근처 장소를 물어봤을 때 사용합니다. 각 장소와 거리를 계산해 가장 가까운 위치를 반환합니다.",
            "func": lambda: tool_set.get_current_location()
        },
        {
            "name": "list_locations",
            "description": "사용자가 어디로 이동할 수 있는지 물어봤을 때 사용합니다. 이동 가능한 장소 목록을 보여줍니다.",
            "func": lambda: ", ".join(tool_set.list_locations())
        },
        {
            "name": "move_to_location",
            "description": "사용자가 특정 장소로 이동하라고 말했을 때 사용합니다.",
            "func": lambda place: tool_set.move_to_location(place)
        },
    ]

    tools = [
        StructuredTool.from_function(
            func=spec["func"],
            name=spec["name"],
            description=spec["description"]
        )
        for spec in tool_specs
    ]

    return tools

class ToolSet(Node):
    def __init__(self, places: dict):
        super().__init__('tool_set')
        self.places = places
        self.frame_id = "map"
        self.base_frame = "base_link"
        self.tolerance_m = 0.15

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        self.ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.goal_handle = None

    def lookup_current_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.frame_id, self.base_frame, Time())
            return (tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.rotation)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def get_current_location(self) -> str:
        cur = self.lookup_current_pose()
        if not cur:
            return "[ERR] 위치 확인 실패"
        cx, cy, _ = cur

        best, dist = None, 1e9
        for name, (x, y, qz, qw) in self.places.items():
            d = math.hypot(cx - x, cy - y)
            if d < dist:
                best, dist = name, d

        if dist < self.tolerance_m:
            return f"[HERE] {best}"
        else:
            return f"[HERE] Unknown (nearest={best}, dist={dist:.2f})"

    def list_locations(self):
        return list(self.places.keys())

    def move_to_location(self, place: str):
        if place not in self.places:
            return f"[ERR] Unknown place: {place}"

        x, y, qz, qw = self.places[place]
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self.frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        goal.pose = ps

        self.ac.wait_for_server()

        self._goal_future = self.ac.send_goal_async(goal)

        def goal_response_callback(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("[NAV] Goal rejected")
                self.goal_handle = None
            else:
                self.get_logger().info("[NAV] Goal accepted")
                self.goal_handle = goal_handle

        self._goal_future.add_done_callback(goal_response_callback)
        return f"[NAV] Heading to {place}"
