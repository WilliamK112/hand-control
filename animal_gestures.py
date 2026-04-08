"""
Hand Gesture → Cute Animal Silhouettes
用手势生成可爱动物剪影
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random


class AnimalGenerator:
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
        
        # 手势 → 动物
        self.gesture_animals = {
            'fist': {
                'name': '🐱 CAT',
                'emoji': '🐱',
                'draw': self.draw_cat,
                'color': (100, 200, 255)
            },
            'open_palm': {
                'name': '🦋 BUTTERFLY',
                'emoji': '🦋',
                'draw': self.draw_butterfly,
                'color': (255, 150, 200)
            },
            'peace': {
                'name': '🐰 RABBIT',
                'emoji': '🐰',
                'draw': self.draw_rabbit,
                'color': (255, 200, 200)
            },
            'point': {
                'name': '🦊 FOX',
                'emoji': '🦊',
                'draw': self.draw_fox,
                'color': (255, 150, 50)
            },
            'ok': {
                'name': '🐸 FROG',
                'emoji': '🐸',
                'draw': self.draw_frog,
                'color': (100, 255, 100)
            },
            'thumbs_up': {
                'name': '🐼 PANDA',
                'emoji': '🐼',
                'draw': self.draw_panda,
                'color': (200, 200, 200)
            },
            'two_up': {
                'name': '🦌 DEER',
                'emoji': '🦌',
                'draw': self.draw_deer,
                'color': (200, 150, 100)
            },
        }
    
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
    
    # 动物绘制函数 - 使用半透明点和线
    def draw_cat(self, frame, cx, cy):
        # 猫 - 用点和线组成
        color = (100, 200, 255)
        # 头部
        cv2.circle(frame, (cx, cy), 60, color, 2)
        # 耳朵
        cv2.line(frame, (cx-40, cy-40), (cx-50, cy-80), color, 2)
        cv2.line(frame, (cx-40, cy-40), (cx-20, cy-70), color, 2)
        cv2.line(frame, (cx+40, cy-40), (cx+50, cy-80), color, 2)
        cv2.line(frame, (cx+40, cy-40), (cx+20, cy-70), color, 2)
        # 眼睛 (点)
        cv2.circle(frame, (cx-20, cy-10), 8, color, -1)
        cv2.circle(frame, (cx+20, cy-10), 8, color, -1)
        # 鼻子
        cv2.circle(frame, (cx, cy+10), 5, color, -1)
        # 嘴巴
        cv2.line(frame, (cx, cy+15), (cx-15, cy+25), color, 2)
        cv2.line(frame, (cx, cy+15), (cx+15, cy+25), color, 2)
        # 胡须
        for i in range(-1, 2):
            cv2.line(frame, (cx-30, cy+5+i*5), (cx-70, cy-10+i*10), color, 1)
            cv2.line(frame, (cx+30, cy+5+i*5), (cx+70, cy-10+i*10), color, 1)
    
    def draw_butterfly(self, frame, cx, cy):
        color = (255, 150, 200)
        # 身体 (点)
        cv2.circle(frame, (cx, cy), 8, color, -1)
        # 翅膀 (椭圆 + 点)
        for dx, dy in [(-80, -30), (-50, -60), (80, -30), (50, -60), (-60, 30), (-30, 60), (60, 30), (30, 60)]:
            cv2.circle(frame, (cx+dx, cy+dy), random.randint(20, 40), color, 2)
        # 翅膀图案 (点)
        for dx, dy in [(-60, -30), (-40, -50), (60, -30), (40, -50), (-40, 30), (40, 30)]:
            cv2.circle(frame, (cx+dx, cy+dy), 5, color, -1)
        # 触角
        cv2.line(frame, (cx-5, cy-8), (cx-20, cy-50), color, 2)
        cv2.line(frame, (cx+5, cy-8), (cx+20, cy-50), color, 2)
    
    def draw_rabbit(self, frame, cx, cy):
        color = (255, 200, 200)
        # 头部
        cv2.circle(frame, (cx, cy), 55, color, 2)
        # 耳朵
        cv2.ellipse(frame, (cx-25, cy-70), (15, 50), 0, 0, 360, color, 2)
        cv2.ellipse(frame, (cx+25, cy-70), (15, 50), 0, 0, 360, color, 2)
        # 眼睛
        cv2.circle(frame, (cx-18, cy-5), 10, color, -1)
        cv2.circle(frame, (cx+18, cy-5), 10, color, -1)
        cv2.circle(frame, (cx-18, cy-5), 4, (50, 50, 50), -1)
        cv2.circle(frame, (cx+18, cy-5), 4, (50, 50, 50), -1)
        # 鼻子
        cv2.circle(frame, (cx, cy+15), 6, color, -1)
        # 嘴巴
        cv2.line(frame, (cx, cy+20), (cx-10, cy+30), color, 2)
        cv2.line(frame, (cx, cy+20), (cx+10, cy+30), color, 2)
    
    def draw_fox(self, frame, cx, cy):
        color = (255, 150, 50)
        # 头
        points = np.array([[cx, cy-50], [cx-50, cy+20], [cx+50, cy+20]], np.int32)
        cv2.polylines(frame, [points], True, color, 2)
        # 耳朵
        cv2.line(frame, (cx-40, cy-20), (cx-50, cy-70), color, 2)
        cv2.line(frame, (cx-50, cy-70), (cx-20, cy-40), color, 2)
        cv2.line(frame, (cx+40, cy-20), (cx+50, cy-70), color, 2)
        cv2.line(frame, (cx+50, cy-70), (cx+20, cy-40), color, 2)
        # 眼睛
        cv2.circle(frame, (cx-15, cy-10), 8, color, -1)
        cv2.circle(frame, (cx+15, cy-10), 8, color, -1)
        cv2.circle(frame, (cx-15, cy-10), 3, (0, 0, 0), -1)
        cv2.circle(frame, (cx+15, cy-10), 3, (0, 0, 0), -1)
        # 鼻子
        cv2.circle(frame, (cx, cy+10), 5, (0, 0, 0), -1)
        # 尾巴
        cv2.ellipse(frame, (cx+80, cy+30), (40, 20), 30, 0, 360, color, 2)
    
    def draw_frog(self, frame, cx, cy):
        color = (100, 255, 100)
        # 身体
        cv2.ellipse(frame, (cx, cy), (70, 60), 0, 0, 360, color, 2)
        # 眼睛 (鼓出来)
        cv2.circle(frame, (cx-25, cy-45), 20, color, 2)
        cv2.circle(frame, (cx+25, cy-45), 20, color, 2)
        cv2.circle(frame, (cx-25, cy-45), 8, (255, 255, 255), -1)
        cv2.circle(frame, (cx+25, cy-45), 8, (255, 255, 255), -1)
        cv2.circle(frame, (cx-25, cy-45), 3, (0, 0, 0), -1)
        cv2.circle(frame, (cx+25, cy-45), 3, (0, 0, 0), -1)
        # 嘴巴
        cv2.arc = cv2.ellipse(frame, (cx, cy+10), (40, 15), 0, 0, 180, color, 2)
        # 鼻子 (点)
        cv2.circle(frame, (cx-10, cy-5), 3, color, -1)
        cv2.circle(frame, (cx+10, cy-5), 3, color, -1)
    
    def draw_panda(self, frame, cx, cy):
        color = (200, 200, 200)
        # 头
        cv2.circle(frame, (cx, cy), 70, color, 2)
        # 耳朵
        cv2.circle(frame, (cx-45, cy-45), 20, color, -1)
        cv2.circle(frame, (cx+45, cy-45), 20, color, -1)
        cv2.circle(frame, (cx-45, cy-45), 10, (50, 50, 50), -1)
        cv2.circle(frame, (cx+45, cy-45), 10, (50, 50, 50), -1)
        # 眼睛
        cv2.circle(frame, (cx-25, cy-10), 15, (255, 255, 255), -1)
        cv2.circle(frame, (cx+25, cy-10), 15, (255, 255, 255), -1)
        cv2.circle(frame, (cx-25, cy-10), 8, (0, 0, 0), -1)
        cv2.circle(frame, (cx+25, cy-10), 8, (0, 0, 0), -1)
        # 鼻子
        cv2.ellipse(frame, (cx, cy+15), (12, 8), 0, 0, 360, (50, 50, 50), -1)
        # 嘴巴
        cv2.arc = cv2.ellipse(frame, (cx, cy+30), (10, 8), 0, 0, 180, color, 2)
    
    def draw_deer(self, frame, cx, cy):
        color = (200, 150, 100)
        # 头
        cv2.circle(frame, (cx, cy), 55, color, 2)
        # 鹿角
        for side in [-1, 1]:
            cv2.line(frame, (cx+side*20, cy-40), (cx+side*40, cy-80), color, 2)
            cv2.line(frame, (cx+side*40, cy-80), (cx+side*20, cy-90), color, 2)
            cv2.line(frame, (cx+side*40, cy-80), (cx+side*50, cy-70), color, 2)
        # 眼睛
        cv2.circle(frame, (cx-18, cy-5), 8, color, -1)
        cv2.circle(frame, (cx+18, cy-5), 8, color, -1)
        cv2.circle(frame, (cx-18, cy-5), 3, (0, 0, 0), -1)
        cv2.circle(frame, (cx+18, cy-5), 3, (0, 0, 0), -1)
        # 鼻子
        cv2.circle(frame, (cx, cy+15), 5, (50, 30, 20), -1)
        # 嘴巴
        cv2.line(frame, (cx, cy+20), (cx-10, cy+30), color, 2)
        cv2.line(frame, (cx, cy+20), (cx+10, cy+30), color, 2)
    
    def draw_hand_with_animal(self, frame, gesture):
        """在手部上方绘制动物"""
        if not self.hand_landmarks:
            return frame
            
        h, w = frame.shape[:2]
        
        # 获取手的位置
        landmarks = self.hand_landmarks[0]
        hand_x = int(landmarks[9].x * w)  # 手掌中心
        hand_y = int(landmarks[9].y * h)
        
        # 获取动物绘制函数
        animal = self.gesture_animals.get(gesture)
        if animal:
            # 在手部上方绘制动物
            animal['draw'](frame, hand_x, hand_y - 150)
        
        # 绘制手部关键点 (半透明)
        color = animal['color'] if animal else (0, 255, 0)
        
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
                cv2.line(frame, pt1, pt2, color, 2)
            
            for landmark in landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame


def main():
    generator = AnimalGenerator()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("🐾 Hand Gesture → Cute Animals")
    print("=" * 50)
    print("手势 → 动物:")
    print("  ✊ fist      → 🐱 CAT")
    print("  ✋ open_palm → 🦋 BUTTERFLY")
    print("  ✌️ peace     → 🐰 RABBIT")
    print("  👆 point     → 🦊 FOX")
    print("  👌 ok        → 🐸 FROG")
    print("  👍 thumbs_up → 🐼 PANDA")
    print("  ✌️ two_up    → 🦌 DEER")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        result = generator.process(frame)
        gesture = generator.get_gesture(0)
        
        if gesture:
            # 绘制手和动物
            frame = generator.draw_hand_with_animal(frame, gesture)
            
            # 显示动物名称
            animal = generator.gesture_animals.get(gesture)
            if animal:
                cv2.putText(frame, animal['name'], (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, animal['color'], 3)
        
        # 标题
        cv2.putText(frame, "🐾 Hand Gesture Animals", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('🐾 Hand Gesture Animals', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()