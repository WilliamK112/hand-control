"""
Hand Gesture DJ Controller - Advanced Version
复杂手势 + 1秒间隔防误触
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import os
import subprocess


class AdvancedGestureDJ:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.05,
            min_tracking_confidence=0.05,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.hand_landmarks = None
        self.volume = 50
        self.last_action_time = 0
        self.last_gesture = None
        self.gesture_confirm_time = 0  # 手势确认时间
        self.min_gesture_duration = 0.3  # 手势需要保持 0.3 秒才确认
        
        # 复杂手势定义（需要更明确的手势）
        self.gesture_commands = {
            # 暂停：拳头（所有手指弯曲）+ 保持 0.5 秒
            'fist': {
                'name': '⏸ PAUSE', 
                'action': 'pause',
                'fingers': {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
            },
            # 播放：手掌完全张开（所有手指伸展）+ 保持 0.5 秒
            'open_palm': {
                'name': '▶ PLAY', 
                'action': 'play',
                'fingers': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True}
            },
            # 下一首：👆 食指向上（只有食指伸展）+ 保持
            'index_up': {
                'name': '⏭ NEXT', 
                'action': 'next',
                'fingers': {'thumb': False, 'index': True, 'middle': False, 'ring': False, 'pinky': False}
            },
            # 上一首：👍 大拇指向上 + 其他手指弯曲
            'thumbs_up': {
                'name': '⏮ PREV', 
                'action': 'prev',
                'fingers': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
            },
            # 音量增大：✋ 手掌 + 向上移动（需要持续移动）
            'volume_up': {
                'name': '🔊 UP', 
                'action': 'volume_up',
                'fingers': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True},
                'require_move': True,
                'direction': 'up'
            },
            # 音量减小：✋ 手掌 + 向下移动
            'volume_down': {
                'name': '🔉 DOWN', 
                'action': 'volume_down',
                'fingers': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True},
                'require_move': True,
                'direction': 'down'
            },
            # 顺时针旋转 → 音量+
            'rotate_cw': {
                'name': '🔊 CW', 
                'action': 'volume_up',
                'fingers': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True}
            },
            # 逆时针旋转 → 音量-
            'rotate_ccw': {
                'name': '🔉 CCW', 
                'action': 'volume_down',
                'fingers': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True}
            },
        }
        
        self.is_playing = False
        self.last_hand_y = 0
        # 旋转检测状态
        self.rotation_history = []  # 存储历史角度
        self.max_rotation_history = 10
        
        print("🎧 Advanced Gesture DJ")
        print("=" * 55)
        print("手势控制:")
        print("  ✊ 拳头         → ⏸ 暂停")
        print("  ✋ 手掌张开     → ▶ 播放")
        print("  👆 食指向上    → ⏭ 下一首")
        print("  👍 大拇指      → ⏮ 上一首")
        print("  🔄 顺时针旋转   → 🔊 音量+")
        print("  🔄 逆时针旋转   → 🔉 音量-")
        print("=" * 55)
        print("🎯 每个手势需保持 0.3 秒触发")
        print("🎯 动作间隔至少 0.5 秒")
        print("=" * 55)
        
    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(frame)
        self.hand_landmarks = result.hand_landmarks
        return result
    
    def get_finger_states(self):
        """获取所有手指状态"""
        if not self.hand_landmarks:
            return None
            
        landmarks = self.hand_landmarks[0]
        
        # 简化检测：指尖 y < PIP y = 伸展
        def get_finger_state(tip, pip):
            return landmarks[tip].y < landmarks[pip].y
        
        return {
            'thumb': get_finger_state(4, 3),
            'index': get_finger_state(8, 7),
            'middle': get_finger_state(12, 11),
            'ring': get_finger_state(16, 15),
            'pinky': get_finger_state(20, 19),
        }
    
    def match_gesture(self, finger_states):
        """匹配手势（带移动检测）"""
        if not finger_states:
            return None
        
        current_time = time.time()
        
        # 检查每种手势
        for gesture_name, config in self.gesture_commands.items():
            target = config['fingers']
            
            # 基础手指匹配
            match = True
            for finger, state in target.items():
                if finger_states.get(finger) != state:
                    match = False
                    break
            
            if not match:
                continue
            
            # 旋转检测：手掌顺时针/逆时针旋转
            if self.hand_landmarks:
                landmarks = self.hand_landmarks[0]
                
                # 获取拇指和食指指尖
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                
                # 计算当前角度
                current_angle = math.atan2(index_tip.y - thumb_tip.y, index_tip.x - thumb_tip.x)
                
                # 记录角度历史
                self.rotation_history.append(current_angle)
                if len(self.rotation_history) > self.max_rotation_history:
                    self.rotation_history.pop(0)
                
                # 需要足够的历史数据
                if len(self.rotation_history) >= 3:
                    # 计算角度变化（考虑角度跳变）
                    angle_changes = []
                    for i in range(1, len(self.rotation_history)):
                        diff = self.rotation_history[i] - self.rotation_history[i-1]
                        # 处理 -pi 到 pi 的跳变
                        if diff > math.pi:
                            diff -= 2 * math.pi
                        elif diff < -math.pi:
                            diff += 2 * math.pi
                        angle_changes.append(diff)
                    
                    total_change = sum(angle_changes)
                    
                    # 顺时针（正角度变化）→ 音量+
                    if total_change > 0.8:  # 约 45 度
                        self.last_gesture = 'rotate_cw'
                        self.gesture_confirm_time = current_time
                        if current_time - self.last_action_time >= 0.3:  # 快速响应
                            return 'rotate_cw'
                    # 逆时针（负角度变化）→ 音量-
                    elif total_change < -0.8:
                        self.last_gesture = 'rotate_ccw'
                        self.gesture_confirm_time = current_time
                        if current_time - self.last_action_time >= 0.3:
                            return 'rotate_ccw'
            
            # 基础手势匹配（不需要移动）
            # 检查是否保持足够时间
            if self.last_gesture == gesture_name:
                if current_time - self.gesture_confirm_time >= self.min_gesture_duration:
                    return gesture_name
            else:
                self.last_gesture = gesture_name
                self.gesture_confirm_time = current_time
            
            return None
        
        # 无匹配，重置
        self.last_gesture = None
        return None
    
    def execute_action(self, action):
        """执行动作（带 1 秒间隔）"""
        now = time.time()
        
        # 检查 1 秒间隔
        if now - self.last_action_time < 0.5:
            return False
            
        self.last_action_time = now
        
        if action == 'play':
            try:
                subprocess.run(['osascript', '-e', 'tell application "Music" to play'], check=True)
                self.is_playing = True
                print("▶ PLAY")
                return True
            except:
                pass
        elif action == 'pause':
            try:
                subprocess.run(['osascript', '-e', 'tell application "Music" to pause'], check=True)
                self.is_playing = False
                print("⏸ PAUSE")
                return True
            except:
                pass
        elif action == 'next':
            try:
                subprocess.run(['osascript', '-e', 'tell application "Music" to next track'], check=True)
                print("⏭ NEXT")
                return True
            except:
                pass
        elif action == 'prev':
            try:
                subprocess.run(['osascript', '-e', 'tell application "Music" to previous track'], check=True)
                print("⏮ PREV")
                return True
            except:
                pass
        elif action == 'volume_up':
            self.volume = min(100, self.volume + 8)
            self.set_volume(self.volume)
            print(f"🔊 VOLUME: {self.volume}%")
            return True
        elif action == 'volume_down':
            self.volume = max(0, self.volume - 8)
            self.set_volume(self.volume)
            print(f"🔉 VOLUME: {self.volume}%")
            return True
            
        return False
    
    def set_volume(self, vol):
        try:
            os.system(f"osascript -e 'set volume output volume {vol}'")
        except:
            pass
    
    def draw_ui(self, frame, gesture, action_triggered):
        """绘制 UI"""
        h, w = frame.shape[:2]
        
        # 背景条
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-140), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 标题
        cv2.putText(frame, "🎧 ADVANCED GESTURE DJ", (25, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # 当前手势
        if gesture:
            cmd = self.gesture_commands.get(gesture, {'name': gesture})
            color = (0, 255, 0) if action_triggered else (0, 150, 255)
            
            # 显示确认进度
            confirm_progress = 0
            if self.last_gesture == gesture:
                confirm_progress = min(1.0, (time.time() - self.gesture_confirm_time) / self.min_gesture_duration)
            
            # 进度条
            if confirm_progress < 1.0 and gesture not in ['volume_up', 'volume_down']:
                bar_w = int(200 * confirm_progress)
                cv2.rectangle(frame, (25, 100), (25 + bar_w, 115), color, -1)
                cv2.rectangle(frame, (25, 100), (225, 115), color, 2)
            
            cv2.putText(frame, cmd['name'], (25, h-95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        
        # 音量条
        bar_w = 250
        bar_h = 25
        bar_x = w - bar_w - 25
        bar_y = h - 65
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        fill = int(bar_w * self.volume / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 255), -1)
        
        cv2.putText(frame, f"🔊 {self.volume}%", (bar_x, bar_y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 状态
        status = "▶ Playing" if self.is_playing else "⏸ Paused"
        status_color = (0, 255, 0) if self.is_playing else (0, 150, 255)
        cv2.putText(frame, status, (w - 160, h-95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # 绘制手部
        if self.hand_landmarks:
            color = (0, 255, 255)
            for landmarks in self.hand_landmarks:
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame


def main():
    dj = AdvancedGestureDJ()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        result = dj.process(frame)
        finger_states = dj.get_finger_states()
        
        gesture = dj.match_gesture(finger_states)
        
        action_triggered = False
        if gesture:
            config = dj.gesture_commands.get(gesture, {})
            action = config.get('action')
            if action:
                action_triggered = dj.execute_action(action)
        
        frame = dj.draw_ui(frame, gesture, action_triggered)
        
        cv2.imshow('🎧 Advanced Gesture DJ', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()