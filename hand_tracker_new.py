"""
Hand Tracking using MediaPipe Tasks Vision API
新的 MediaPipe API (0.10+)
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class HandTracker:
    def __init__(
        self,
        num_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ):
        # 配置 MediaPipe Hands
        base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # 21 个关键点
        self.hand_landmarks = None
        self.handedness = None
        
    def process(self, frame_bgr):
        """处理帧，返回手部关键点"""
        # MediaPipe expects RGB input
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Use frame count as simple timestamp
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        timestamp_ms = self._frame_count * 33  # ~30fps
        
        result = self.detector.detect_for_video(frame, timestamp_ms)
        
        self.hand_landmarks = result.hand_landmarks
        self.handedness = result.handedness
        
        return result
        
    def get_finger_state(self, hand_idx=0):
        """获取手指状态（伸直/弯曲）"""
        if not self.hand_landmarks or hand_idx >= len(self.hand_landmarks):
            return {}
            
        landmarks = self.hand_landmarks[hand_idx]
        
        # 关键点索引
        # Thumb: 1-4
        # Index: 5-8
        # Middle: 9-12
        # Ring: 13-16
        # Pinky: 17-20
        
        def is_extended(tip_idx, pip_idx):
            """判断手指是否伸直"""
            return landmarks[tip_idx].y < landmarks[pip_idx].y
            
        return {
            'thumb': is_extended(4, 3),
            'index': is_extended(8, 7),
            'middle': is_extended(12, 11),
            'ring': is_extended(16, 15),
            'pinky': is_extended(20, 19),
        }
    
    def get_index_finger_tip(self, hand_idx=0):
        """获取食指指尖位置"""
        if not self.hand_landmarks or hand_idx >= len(self.hand_landmarks):
            return None
        return self.hand_landmarks[hand_idx][8]
    
    def draw_landmarks(self, frame):
        """绘制手部关键点"""
        if not self.hand_landmarks:
            return frame
            
        h, w = frame.shape[:2]
        
        for hand_idx, landmarks in enumerate(self.hand_landmarks):
            # 绘制连接线
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),   # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]
            
            for start, end in connections:
                pt1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
                pt2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # 绘制关键点
            for landmark in landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
        return frame


def main():
    tracker = HandTracker()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("按 'q' 或 'Esc' 退出")
    print("按 'h' 切换显示关键点")
    
    show_overlay = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 水平翻转（镜像）
        frame = cv2.flip(frame, 1)
        
        # 处理帧
        result = tracker.process(frame)
        
        # 获取手指状态
        finger_state = tracker.get_finger_state(0)
        if finger_state:
            # 在屏幕上显示手指状态
            status = " | ".join([f"{k}: {v}" for k, v in finger_state.items()])
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 绘制关键点
        if show_overlay:
            frame = tracker.draw_landmarks(frame)
            
        # 显示
        cv2.imshow('Hand Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('h'):
            show_overlay = not show_overlay
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
