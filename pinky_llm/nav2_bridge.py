import math
import time
import rclpy
import requests
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time
from datetime import date, datetime
from typing import List, Dict
from langchain_teddynote.tools import GoogleNews
from pinky_llm.object_detector import ObjectDetector

CITY_MAP = {
    "Busan": "부산",
    "Seoul": "서울",
    "Incheon": "인천",
    "Daegu": "대구",
    "Gwangju": "광주",
    "Daejeon": "대전",
    "Ulsan": "울산",
    "Jeju": "제주",
}

class Nav2Bridge(Node):
    def __init__(self, places: dict):
        super().__init__('nav2_bridge')
        self._places = places
        self._frame_id = "map"
        self._base_frame = "base_link"
        self._tolerance_m = 0.15

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)

        self._ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # self.detect_client = self.create_client(ObjectDetect, '/detect_once')
        # while not self.detect_client.wait_for_service(timeout_sec=2.0):
        #     self.get_logger().info("Waiting for object_detector service...")

        self._places = places
        self.detector = ObjectDetector(
            model_path="./yolo11n.pt",
            conf=0.5,
            stream_url="http://192.168.4.1:5000/camera"
        )


        self._goal_handle = None
            
    def _lookup_current_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(self._frame_id, self._base_frame, Time())
            return (tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.rotation)   # orientation은 Quaternion
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def list_places(self):
        return list(self._places.keys())

    def where_am_i(self):
        cur = self._lookup_current_pose()
        if not cur:
            return "[ERR] 위치 확인 실패"
        cx, cy, _ = cur
        best, dist = None, 1e9
        for name, (x,y,qz,qw) in self._places.items():
            d = math.hypot(cx-x, cy-y)
            if d < dist:
                best, dist = name, d
        if dist < self._tolerance_m:
            return f"[HERE] {best}"
        else:
            return f"[HERE] Unknown (nearest={best}, dist={dist:.2f})"

    def go_to(self, place: str):
        if place not in self._places:
            return f"[ERR] Unknown place: {place}"
        x,y,qz,qw = self._places[place]
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self._frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        goal.pose = ps
        
        self._ac.wait_for_server()
        
        self._goal_future = self._ac.send_goal_async(goal)
        def goal_response_callback(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("[NAV] Goal rejected")
                self._goal_handle = None
            else:
                self.get_logger().info("[NAV] Goal accepted")
                self._goal_handle = goal_handle

        self._goal_future.add_done_callback(goal_response_callback)
        return f"[NAV] Heading to {place}"
        
        
        # self._ac.send_goal_async(goal)
        # return f"[NAV] Heading to {place}"

    def cancel_goal(self):
        if hasattr(self, "_goal_handle") and self._goal_handle is not None:
            cancel_future = self._goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future)  # 동기 대기
            cancel_response = cancel_future.result()
            if len(cancel_response.goals_canceling) > 0:
                return "[NAV] Goal successfully canceled"
            else:
                return "[NAV] No goal was canceled"
        else:
            return "[NAV] No active goal"

    def turn_around(self, ang_speed: float = -1.57, duration: float = 4.0):
        self.cancel_goal()
        twist = Twist()
        twist.angular.z = ang_speed
        end_time = self.get_clock().now() + Duration(seconds=duration)

        while self.get_clock().now() < end_time:
            self._cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(0.1)

        stop_twist = Twist()
        for _ in range(5):
            self._cmd_pub.publish(stop_twist)
            time.sleep(0.05)
        return f"[SPIN] 제자리 한바퀴 회전"

    def go_forward_a_little(self, lin_speed: float = 0.1, duration: float = 1.0):
        self.cancel_goal()
        twist = Twist()
        twist.linear.x = lin_speed
        end_time = self.get_clock().now() + Duration(seconds=duration)

        while self.get_clock().now() < end_time:
            self._cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(0.1)

        # stop 신호를 안정적으로 여러 번 발행
        stop_twist = Twist()
        for _ in range(5):
            self._cmd_pub.publish(stop_twist)
            time.sleep(0.05)
        return f"[MOVE] 조금 전진"
    
    def go_backward_a_little(self, lin_speed: float = -0.1, duration: float = 1.0):
        self.cancel_goal()
        twist = Twist()
        twist.linear.x = lin_speed
        end_time = self.get_clock().now() + Duration(seconds=duration)

        while self.get_clock().now() < end_time:
            self._cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(0.1)

        # stop 신호를 안정적으로 여러 번 발행
        stop_twist = Twist()
        for _ in range(5):
            self._cmd_pub.publish(stop_twist)
            time.sleep(0.05)
        return f"[MOVE] 조금 후진"
    
    def turn_left_a_little(self, ang_speed: float = 0.524, duration: float = 1.0):
        self.cancel_goal()
        twist = Twist()
        twist.angular.z = ang_speed
        end_time = self.get_clock().now() + Duration(seconds=duration)

        while self.get_clock().now() < end_time:
            self._cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(0.1)

        stop_twist = Twist()
        for _ in range(5):
            self._cmd_pub.publish(stop_twist)
            time.sleep(0.05)
        return f"[MOVE] 왼쪽으로 조금 회전"
    
    def turn_right_a_little(self, ang_speed: float = -0.524, duration: float = 1.0):
        self.cancel_goal()
        twist = Twist()
        twist.angular.z = ang_speed
        end_time = self.get_clock().now() + Duration(seconds=duration)

        while self.get_clock().now() < end_time:
            self._cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(0.1)

        stop_twist = Twist()
        for _ in range(5):
            self._cmd_pub.publish(stop_twist)
            time.sleep(0.05)
        return f"[MOVE] 오른쪽으로 조금 회전"
    
    def lets_dance(self):
        self.cancel_goal()
        seq = [
            ( 1.048, 1.0),   # 좌로 0.5초
            (-1.048, 2.0),   # 우로 1.0초
            ( 1.048, 2.0),   # 좌로 1.0초
            (-1.048, 1.0),   # 우로 0.5초
        ]

        for ang_speed, duration in seq:
            twist = Twist()
            twist.angular.z = ang_speed
            end_time = self.get_clock().now() + Duration(seconds=duration)

            while self.get_clock().now() < end_time:
                self._cmd_pub.publish(twist)
                rclpy.spin_once(self, timeout_sec=0)

        stop_twist = Twist()
        for _ in range(5):
            self._cmd_pub.publish(stop_twist)
            time.sleep(0.05)

        return "[DANCE] 좌우로 흔들며 춤을 춥니다."
    
    def what_time_is_it_now(self) -> str:
        self.cancel_goal()
        return datetime.now().strftime("%H시%M분")
    
    def what_date_today(self) -> str:
        self.cancel_goal()
        weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        return f"{str(date.today())} ({weekdays[date.today().weekday()]})"
    
    def whats_the_weather(self, city: str = "Busan"):
        self.cancel_goal()
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=2fa868796bd69f60adf7498fb64db09f&units=metric&lang=kr"
        resp = requests.get(url)
        
        if resp.status_code != 200:
            return f"[ERR] 날씨 정보를 불러올 수 없음 (status {resp.status_code})"
        
        data = resp.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"{CITY_MAP.get(city)}의 현재 날씨: {weather}, 기온 {int(temp)}°C"
    
    def summarizing_news(self, search_keyword: str):
        self.cancel_goal()
        news_tool = GoogleNews()
        results = news_tool.search_by_keyword(search_keyword, k=5)
        return results

    def get_detect_info(self) -> str:
        self.cancel_goal()
        return self.detector.get_detect_info()