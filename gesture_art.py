"""
Hand Gesture AI Art Generator - WITH VISUAL EFFECTS
用手势控制实时视觉特效
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random


class GestureArtGenerator:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.hand_landmarks = None
        self.handedness = None
        
        # 手势到视觉效果
        self.gesture_effects = {
            'fist': {'name': '🔥 ABSTRACT GEOMETRIC', 'color': (255, 50, 50), 'pattern': 'geometric'},
            'open_palm': {'name': '🌊 WATERCOLOR DREAM', 'color': (100, 200, 255), 'pattern': 'watercolor'},
            'peace': {'name': '🌆 CYBERPUNK', 'color': (255, 0, 255), 'pattern': 'neon'},
            'point': {'name': '🎭 CLASSICAL PORTRAIT', 'color': (255, 200, 100), 'pattern': 'classic'},
            'ok': {'name': '🌲 FANTASY FOREST', 'color': (50, 200, 50), 'pattern': 'nature'},
            'thumbs_up': {'name': '🎸 80s SYNTHWAVE', 'color': (255, 100, 0), 'pattern': 'retro'},
            'two_up': {'name': '🎨 ABSTRACT EXPRESSION', 'color': (200, 50, 200), 'pattern': 'splash'},
        }
        
        self.current_effect = None
        self.particles = []
        
    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect_for_video(frame, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000))
        self.hand_landmarks = result.hand_landmarks
        return result
    
    def get_gesture(self, hand_idx=0):
        if not self.hand_landmarks or hand_idx >= len(self.hand_landmarks):
            return None
            
        landmarks = self.hand_landmarks[hand_idx]
        
        def get_finger_state(tip, pip):
            return landmarks[tip].y < landmarks[pip].y
        
        thumb = get_finger_state(4, 3)
        index = get_finger_state(8, 7)
        middle = get_finger_state(12, 11)
        ring = get_finger_state(16, 15)
        pinky = get_finger_state(20, 19)
        
        if not thumb and not index and not middle and not ring and not pinky:
            return 'fist'
        elif thumb and index and middle and ring and pinky:
            return 'open_palm'
        elif index and middle and not thumb and not ring and not pinky:
            return 'peace'
        elif index and not middle and not ring and not pinky:
            return 'point'
        elif thumb and index:
            dist = ((landmarks[4].x - landmarks[8].x)**2 + (landmarks[4].y - landmarks[8].y)**2)**0.5
            if dist < 0.08:
                return 'ok'
        elif not index and not middle and not ring and not pinky and thumb:
            return 'thumbs_up'
        elif index and middle and not ring and not pinky:
            return 'two_up'
            
        return None
    
    def apply_visual_effect(self, frame, gesture):
        """根据手势应用不同的视觉效果"""
        h, w = frame.shape[:2]
        
        if gesture not in self.gesture_effects:
            return frame
            
        effect = self.gesture_effects[gesture]
        self.current_effect = effect
        color = effect['color']
        pattern = effect['pattern']
        
        # 应用不同效果
        if pattern == 'geometric':
            # 抽象几何 - 添加几何形状
            for _ in range(5):
                x, y = random.randint(0, w), random.randint(0, h)
                size = random.randint(30, 100)
                shape = random.choice([cv2.RECTANGLE, cv2.ELLIPSE])
                if shape == cv2.RECTANGLE:
                    cv2.rectangle(frame, (x, y), (x+size, y+size), color, -1)
                else:
                    cv2.ellipse(frame, (x, y), (size, size//2), 0, 0, 360, color, -1)
                    
        elif pattern == 'watercolor':
            # 水彩效果 - 模糊和透明度
            kernel = np.ones((30, 30), np.float32) / 900
            blurred = cv2.filter2D(frame, -1, kernel)
            frame = cv2.addWeighted(frame, 0.7, blurred, 0.3, 0)
            # 添加水彩斑块
            for _ in range(3):
                x, y = random.randint(0, w-100), random.randint(0, h-100)
                cv2.circle(frame, (x, y), random.randint(50, 150), color, -1)
                
        elif pattern == 'neon':
            # 赛博朋克 - 霓虹线条
            for i in range(10):
                x1, y1 = random.randint(0, w), random.randint(0, h)
                x2, y2 = x1 + random.randint(-200, 200), y1 + random.randint(-200, 200)
                cv2.line(frame, (x1, y1), (x2, y2), color, random.randint(2, 5))
            # 添加扫描线
            for y in range(0, h, 4):
                cv2.line(frame, (0, y), (w, y), (0, 0, 0), 1)
                
        elif pattern == 'classic':
            # 古典油画 - 暖色调滤镜
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_AUTUMN)
            # 添加暗角
            vignette = np.zeros((h, w))
            for i in range(h):
                for j in range(w):
                    dist = ((i - h/2)**2 + (j - w/2)**2)**0.5 / ((h**2 + w**2)**0.5 / 2)
                    vignette[i, j] = max(0, 1 - dist)
            vignette = np.dstack([vignette] * 3)
            frame = (frame * vignette).astype(np.uint8)
            
        elif pattern == 'nature':
            # 自然风景 - 绿色滤镜 + 圆形
            frame = cv2.add(frame, np.array([0, 30, 0]))
            for _ in range(8):
                x, y = random.randint(0, w), random.randint(0, h)
                cv2.circle(frame, (x, y), random.randint(20, 80), color, -1)
                
        elif pattern == 'retro':
            # 80s 复古 - 紫橙渐变
            gradient = np.zeros((h, w, 3))
            gradient[:, :, 0] = np.linspace(100, 255, w).reshape(1, -1)  # 红
            gradient[:, :, 1] = np.linspace(0, 100, w).reshape(1, -1)     # 绿
            gradient[:, :, 2] = np.linspace(150, 255, w).reshape(1, -1)   # 蓝
            frame = cv2.addWeighted(frame, 0.6, gradient.astype(np.uint8), 0.4, 0)
            # 网格线
            for i in range(0, h, 40):
                cv2.line(frame, (0, i), (w, i), (255, 0, 255), 1)
            for j in range(0, w, 40):
                cv2.line(frame, (j, 0), (j, h), (255, 0, 255), 1)
                
        elif pattern == 'splash':
            # 抽象表现主义 - 飞溅效果
            for _ in range(20):
                x, y = random.randint(0, w), random.randint(0, h)
                cv2.circle(frame, (x, y), random.randint(5, 30), color, -1)
            # 随机线条
            for _ in range(5):
                pts = [(random.randint(0, w), random.randint(0, h)) for _ in range(3)]
                cv2.polylines(frame, [np.array(pts)], False, color, 3)
        
        return frame
    
    def draw_landmarks(self, frame, gesture):
        """绘制手部关键点"""
        if not self.hand_landmarks:
            return frame
            
        h, w = frame.shape[:2]
        
        effect = self.gesture_effects.get(gesture, {'color': (0, 255, 0)})
        color = effect['color']
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for landmarks in self.hand_landmarks:
            for start, end in connections:
                pt1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
                pt2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
                cv2.line(frame, pt1, pt2, color, 4)
            
            for i, landmark in enumerate(landmarks):
                x, y = int(landmark.x * w), int(landmark.y * h)
                if i == 8:  # 食指指尖
                    cv2.circle(frame, (x, y), 15, (255, 255, 255), -1)
                    cv2.circle(frame, (x, y), 10, color, -1)
                else:
                    cv2.circle(frame, (x, y), 6, color, -1)
        
        return frame


def main():
    generator = GestureArtGenerator()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("🎨 Hand Gesture AI Art - VISUAL EDITION")
    print("=" * 50)
    print("手势 → 视觉效果:")
    print("  ✊ fist      → 抽象几何 (红)")
    print("  ✋ open_palm → 水彩梦幻 (蓝)")
    print("  ✌️ peace     → 赛博朋克 (紫)")
    print("  👆 point     → 古典肖像 (金)")
    print("  👌 ok        → 奇幻森林 (绿)")
    print("  👍 thumbs_up → 80s 复古 (橙)")
    print("  ✌️ two_up    → 抽象表现 (紫)")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        result = generator.process(frame)
        gesture = generator.get_gesture(0)
        
        if gesture:
            # 应用视觉效果
            frame = generator.apply_visual_effect(frame, gesture)
            # 绘制关键点
            frame = generator.draw_landmarks(frame, gesture)
            
            # 显示当前效果
            effect_name = generator.gesture_effects[gesture]['name']
            cv2.putText(frame, effect_name, (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # 标题
        cv2.putText(frame, "🎨 Hand Gesture AI Art", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Hand Gesture AI Art', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()