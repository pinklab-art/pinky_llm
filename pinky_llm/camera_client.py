import cv2
import requests
import numpy as np

def main():
    url = 'http://192.168.4.1:5000/camera'
    
    stream = requests.get(url, stream=True)
    bytes_data = b''

    cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Feed", 800, 600)

    try:
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            
            # JPEG 시작(FFD8)과 끝(FFD9) 마커를 찾아서 프레임 추출
            start = bytes_data.find(b'\xff\xd8')
            end = bytes_data.find(b'\xff\xd9')
            
            if start != -1 and end != -1:
                jpg = bytes_data[start:end+2]  # JPEG 이미지 데이터 추출
                bytes_data = bytes_data[end+2:]  
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow("Video Feed", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()