"""
Hand Gesture → 3D Artistic Skeleton
半透明白色 3D 骨架艺术
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import random


class ArtisticSkeleton:
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
        
        # 手势 → 骨架风格
        self.gesture_styles = {
            'fist': {
                'name': 'DIAMOND SKELETON',
                'z_depth': 0.8,
                'complexity': 'high',
                '3d_effect': True
            },
            'open_palm': {
                'name': 'STAR CONSTELLATION',
                'z_depth': 1.2,
                'complexity': 'very_high',
                '3d_effect': True
            },
            'peace': {
                'name': 'TWIN TOWERS',
                'z_depth': 1.0,
                'complexity': 'medium',
                '3d_effect': True
            },
            'point': {
                'name': 'LASER GRID',
                'z_depth': 0.6,
                'complexity': 'high',
                '3d_effect': True
            },
            'ok': {
                'name': 'RING PORTAL',
                'z_depth': 1.5,
                'complexity': 'high',
                '3d_effect': True
            },
            'thumbs_up': {
                'name': 'PYRAMID SPIRE',
                'z_depth': 1.3,
                'complexity': 'medium',
                '3d_effect': True
            },
            'two_up': {
                'name': 'CATHEDRAL ARCH',
                'z_depth': 1.1,
                'complexity': 'high',
                '3d_effect': True
            },
        }
        
        self.particles = []
        
    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect_for_video(frame, self.frame_count * 33)
        self.hand_landmarks = result.hand_landmarks
        
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        
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
    
    def draw_3d_skeleton(self, frame, gesture, landmarks):
        """绘制 3D 艺术骨架"""
        h, w = frame.shape[:2]
        
        style = self.gesture_styles.get(gesture, {'z_depth': 1.0, 'complexity': 'medium'})
        z_depth = style['z_depth']
        
        # 白色半透明颜色 (B, G, A) - 透明度可调
        white = (255, 255, 255)
        white_transparent = (255, 255, 200)  # 半透明白
        
        # 3D 效果：添加深度偏移的点
        def get_3d_point(x, y, z_offset=0, index=0):
            # 基于索引和深度创建 3D 偏移
            depth = z_offset * (1 + index * 0.05)
            offset_x = int(depth * 20 * math.sin(index * 0.5))
            offset_y = int(depth * 15 * math.cos(index * 0.3))
            return int(x * w) + offset_x, int(y * h) + offset_y
        
        # 主连接线 (21 个关键点完整连接)
        connections = [
            # 手腕到手指
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            # 手掌连接
            (5, 9), (9, 13), (13, 17),  # 横跨手掌
            # 额外 3D 连接 (创建更复杂的骨架)
            (0, 8), (0, 12), (0, 16), (0, 20),  # 手腕到指尖对角线
            (4, 8), (8, 12), (12, 16), (16, 20),  # 指尖之间的连接
        ]
        
        # 绘制多层骨架（创造深度感）
        for layer in range(3):
            layer_offset = (layer - 1) * 0.1 * z_depth
            alpha = 255 - layer * 60  # 透明度
            
            for start, end in connections:
                if start >= len(landmarks) or end >= len(landmarks):
                    continue
                    
                pt1 = get_3d_point(landmarks[start].x, landmarks[start].y, layer_offset, start)
                pt2 = get_3d_point(landmarks[end].x, landmarks[end].y, layer_offset, end)
                
                # 线条宽度随层级变化
                thickness = 3 - layer
                if thickness < 1:
                    thickness = 1
                    
                cv2.line(frame, pt1, pt2, white_transparent, thickness)
        
        # 绘制 3D 关键点（多层圆点）
        for i, landmark in enumerate(landmarks):
            for layer in range(3):
                layer_offset = (layer - 1) * 0.08 * z_depth
                x, y = get_3d_point(landmark.x, landmark.y, layer_offset, i)
                
                # 圆点大小随索引变化（指尖更大）
                if i in [4, 8, 12, 16, 20]:  # 指尖
                    radius = 12 - layer * 3
                elif i == 0:  # 手腕
                    radius = 10 - layer * 2
                else:
                    radius = 8 - layer * 2
                    
                if radius < 2:
                    radius = 2
                    
                # 绘制空心圆
                cv2.circle(frame, (x, y), radius, white_transparent, 2)
        
        # 添加复杂装饰元素
        if style['complexity'] in ['high', 'very_high']:
            # 添加星座式的小点连接
            num_extra = 30 if style['complexity'] == 'very_high' else 15
            for _ in range(num_extra):
                # 随机生成围绕手部的点
                center_x = landmarks[9].x  # 手掌中心
                center_y = landmarks[9].y
                
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(0.05, 0.3) * z_depth
                
                px = center_x + dist * math.cos(angle)
                py = center_y + dist * math.sin(angle)
                
                px_int = int(px * w)
                py_int = int(py * h)
                
                # 随机小点
                size = random.randint(1, 4)
                cv2.circle(frame, (px_int, py_int), size, white_transparent, -1)
                
                # 随机连线到最近的关键点
                if random.random() > 0.7:
                    min_dist = float('inf')
                    closest = None
                    for lm in landmarks:
                        d = ((lm.x - px)**2 + (lm.y - py)**2)**0.5
                        if d < min_dist:
                            min_dist = d
                            closest = lm
                    if closest:
                        pt2 = (int(closest.x * w), int(closest.y * h))
                        cv2.line(frame, (px_int, py_int), pt2, white_transparent, 1)
        
        # 添加 3D 环绕效果
        if z_depth > 1.0:
            center_x = int(landmarks[9].x * w)
            center_y = int(landmarks[9].y * h)
            radius = int(150 * z_depth)
            
            # 绘制同心圆环
            for r in range(50, radius, 30):
                alpha = int(255 * (1 - r / radius) * 0.3)
                color = (255, 255, alpha)
                cv2.circle(frame, (center_x, center_y), r, color, 1)
        
        return frame
    
    def draw_hand(self, frame, gesture):
        """绘制完整的手势艺术"""
        if not self.hand_landmarks:
            return frame
            
        for landmarks in self.hand_landmarks:
            frame = self.draw_3d_skeleton(frame, gesture, landmarks)
        
        return frame


def main():
    generator = ArtisticSkeleton()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✨ 3D Artistic Skeleton")
    print("=" * 60)
    print("手势 → 骨架风格:")
    print("  ✊ fist      → 💎 DIAMOND SKELETON")
    print("  ✋ open_palm → ⭐ STAR CONSTELLATION")
    print("  ✌️ peace     → 🗼 TWIN TOWERS")
    print("  👆 point     → 🎯 LASER GRID")
    print("  👌 ok        → 🌀 RING PORTAL")
    print("  👍 thumbs_up → 🔺 PYRAMID SPIRE")
    print("  ✌️ two_up    → 🏛️ CATHEDRAL ARCH")
    print("=" * 60)
    print("白色半透明 + 3D 深度效果 + 复杂骨架")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        result = generator.process(frame)
        gesture = generator.get_gesture(0)
        
        if gesture:
            frame = generator.draw_hand(frame, gesture)
            
            # 显示风格名称
            style = generator.gesture_styles.get(gesture)
            if style:
                cv2.putText(frame, style['name'], (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # 标题
        cv2.putText(frame, "✨ 3D Artistic Skeleton", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('3D Artistic Skeleton', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()