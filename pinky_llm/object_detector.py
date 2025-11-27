import cv2
import requests
import numpy as np
import threading
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="./yolo11n.pt", conf=0.5, stream_url="http://192.168.4.1:5000/camera"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.stream_url = stream_url
        self.bytes_data = b''
        self.latest_frame = None
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # self.stream = requests.get(self.stream_url, stream=True)

    def _capture_loop(self):
        stream = requests.get(self.stream_url, stream=True)
        for chunk in stream.iter_content(chunk_size=1024):
            self.bytes_data += chunk
            start = self.bytes_data.find(b'\xff\xd8')
            end = self.bytes_data.find(b'\xff\xd9')
            if start != -1 and end != -1:
                jpg = self.bytes_data[start:end+2]
                self.bytes_data = self.bytes_data[end+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def detect(self, frame: np.ndarray) -> str:
        """YOLO로 객체 검출 후 방향 정보 반환"""
        results = self.model(frame, conf=self.conf, device="cpu", verbose=False)
        w = frame.shape[1]

        info = {}
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                label = self.model.names[cls]
                x, _, _, _ = map(int, box.xywh[0])
                confidence = round(box.conf[0].item(), 2)

                if x < w / 3:
                    direction = "왼쪽"
                elif x > w * 2 / 3:
                    direction = "오른쪽"
                else:
                    direction = "중앙"
                if (label not in info) or (confidence > info[label].get("confidence", 0)):
                    info[label] = {"label": label, "direction": direction}

        if not info:
            return "감지된 객체 없음"
        return ", ".join(f"{v['label']}:{v['direction']}" for v in info.values())

    def get_detect_info(self) -> str:
        frame = self.get_frame()
        if frame is None:
            return "[ERR] 프레임 없음"
        try:
            return self.detect(frame)
        except Exception as e:
            return f"[ERR] YOLO detect 실패: {e}"