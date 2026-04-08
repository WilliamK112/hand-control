"""
3D Artistic Music Player
手势 + 键盘控制 Apple Music app
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import random
import subprocess
import time


class AppleMusicController:
    def __init__(self):
        # MediaPipe hand tracking
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
        self.frame_count = 0
        
        # Visualizer settings
        self.particles = []
        
        # Gesture to action mapping
        self.gesture_actions = {
            'fist': ('pause', '⏸ PAUSE'),
            'open_palm': ('play', '▶ PLAY'),
            'peace': ('next', '⏭ NEXT'),
            'point': ('visual', '🎨 VISUAL'),
            'ok': ('volume_up', '🔊 UP'),
            'thumbs_up': ('volume_down', '🔈 DOWN'),
            'two_up': ('random', '🎲 RANDOM'),
        }
        
        # Current visual mode
        self.visual_mode = 0
        self.visual_modes = ['WAVE', 'PARTICLE', 'SPIRAL', 'GALAXY']
        
        self.last_gesture = None
        self.last_action_time = 0
        self.action_cooldown = 0.5  # 0.5 second cooldown for responsive control
        
        # Get initial state
        self.is_playing = self._get_player_state()
        self.current_track = self._get_current_track()
        
    def _run_applescript(self, script):
        """Run AppleScript command"""
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout.strip()
        except:
            return None
    
    def _get_player_state(self):
        """Get Music app player state"""
        state = self._run_applescript('tell application "Music" to get player state')
        return state == 'playing'
    
    def _get_current_track(self):
        """Get current track name"""
        return self._run_applescript('tell application "Music" to get name of current track')
    
    def play(self):
        """Play music"""
        self._run_applescript('tell application "Music" to play')
        self.is_playing = True
        
    def pause(self):
        """Pause music"""
        self._run_applescript('tell application "Music" to pause')
        self.is_playing = False
        
    def next_track(self):
        """Next track"""
        self._run_applescript('tell application "Music" to next track')
        time.sleep(0.5)
        self.current_track = self._get_current_track()
        
    def volume_up(self):
        """Volume up"""
        self._run_applescript('tell application "Music" to set sound volume to sound volume + 10')
        
    def volume_down(self):
        """Volume down"""
        self._run_applescript('tell application "Music" to set sound volume to sound volume - 10')
        
    def process(self, frame_bgr):
        self.frame_count += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)  # IMAGE mode for better detection
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
    
    def handle_gesture(self, gesture):
        """Handle gesture-based controls"""
        current_time = time.time()
        
        if gesture == self.last_gesture and (current_time - self.last_action_time) < self.action_cooldown:
            return None
            
        if gesture and gesture in self.gesture_actions:
            self.last_gesture = gesture
            self.last_action_time = current_time
            return self.gesture_actions[gesture]
        
        return None
    
    def execute_action(self, action):
        """Execute action"""
        if action == 'play':
            self.play()
        elif action == 'pause':
            self.pause()
        elif action == 'next':
            self.next_track()
        elif action == 'volume_up':
            self.volume_up()
        elif action == 'volume_down':
            self.volume_down()
        elif action == 'visual':
            self.visual_mode = (self.visual_mode + 1) % len(self.visual_modes)
        elif action == 'random':
            self.visual_mode = random.randint(0, len(self.visual_modes) - 1)
    
    def get_volume(self):
        """Get current volume"""
        vol = self._run_applescript('tell application "Music" to get sound volume')
        try:
            return int(vol) / 100.0
        except:
            return 0.5
    
    def draw_visualization(self, frame):
        """Draw music visualization"""
        h, w = frame.shape[:2]
        
        mode = self.visual_modes[self.visual_mode]
        
        if mode == 'WAVE':
            # Audio wave at bottom
            bottom_y = h - 100
            for i in range(0, w, 4):
                amplitude = int(40 * math.sin(i * 0.03 + self.frame_count * 0.1))
                cv2.line(frame, (i, bottom_y), (i, bottom_y + amplitude), (100, 200, 255), 2)
                
        elif mode == 'PARTICLE':
            # Particle system - subtle
            if random.random() > 0.5:
                self.particles.append({
                    'x': random.randint(0, w),
                    'y': random.randint(0, h),
                    'vx': random.uniform(-2, 2),
                    'vy': random.uniform(-2, 2),
                    'life': 80,
                    'color': (random.randint(150, 255), random.randint(150, 255), random.randint(200, 255))
                })
            
            for p in self.particles[:]:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['life'] -= 2
                
                if p['life'] <= 0:
                    self.particles.remove(p)
                else:
                    cv2.circle(frame, (int(p['x']), int(p['y'])), int(p['life'] / 20), p['color'], -1)
                    
        elif mode == 'SPIRAL':
            # Subtle spiral in corners
            corners = [(50, 50), (w-50, 50), (50, h-50), (w-50, h-50)]
            for cx, cy in corners:
                for i in range(30):
                    angle = i * 0.2 + self.frame_count * 0.03
                    radius = i * 2
                    x = int(cx + radius * math.cos(angle))
                    y = int(cy + radius * math.sin(angle))
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), 1, (200, 150, 255), -1)
                    
        elif mode == 'GALAXY':
            # Subtle galaxy in corners
            corners = [(w//4, h//4), (3*w//4, h//4), (w//4, 3*h//4), (3*w//4, 3*h//4)]
            for cx, cy in corners:
                for i in range(50):
                    angle = i * 0.1 + self.frame_count * 0.01
                    radius = i * 1.5
                    x = int(cx + radius * math.cos(angle))
                    y = int(cy + radius * math.sin(angle))
                    if 0 <= x < w and 0 <= y < h:
                        color = (min(200, i * 3), max(100, 200 - i * 2), min(255, 150 + i))
                        cv2.circle(frame, (x, y), 1, color, -1)
        
        return frame
    
    def draw_hand_skeleton(self, frame, landmarks):
        """Draw artistic skeleton"""
        h, w = frame.shape[:2]
        white = (255, 255, 255)
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]
        
        for start, end in connections:
            if start >= len(landmarks) or end >= len(landmarks):
                continue
            pt1 = (int(landmarks[start].x * w), int(landmarks[start].y * h))
            pt2 = (int(landmarks[end].x * w), int(landmarks[end].y * h))
            cv2.line(frame, pt1, pt2, white, 2)
        
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, white, -1)
        
        return frame


def main():
    player = AppleMusicController()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "=" * 60)
    print("🎵 3D Artistic Music Player (Apple Music)")
    print("=" * 60)
    print("手势控制:")
    print("  ✊ fist      → ⏸ 暂停")
    print("  ✋ open_palm → ▶ 播放")
    print("  ✌️ peace     → ⏭ 下一首")
    print("  👆 point     → 🎨 切换视觉模式")
    print("  👌 ok        → 🔊 增加音量")
    print("  👍 thumbs_up → 🔈 降低音量")
    print("=" * 60)
    print("键盘控制: SPACE=播放/暂停, N=下一首, V=模式, +/-=音量")
    print("=" * 60)
    
    current_action = None
    action_display_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        result = player.process(frame)
        gesture = player.get_gesture(0)
        
        # Handle gesture
        action = player.handle_gesture(gesture)
        if action:
            player.execute_action(action)
            current_action = action[1]
            action_display_time = time.time()
        
        # Draw hand skeleton
        if player.hand_landmarks:
            for landmarks in player.hand_landmarks:
                frame = player.draw_hand_skeleton(frame, landmarks)
        
        # Get frame dimensions for text positioning
        h, w = frame.shape[:2]
        track = player.current_track or "No track"
        
        # Draw visualization (subtle, at edges)
        frame = player.draw_visualization(frame)
        
        # Display mode (subtle, bottom left)
        cv2.putText(frame, f"Mode: {player.visual_modes[player.visual_mode]}", (20, h-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.putText(frame, f"♪ {track[:40]}", (20, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 150, 100), 1)
        
        # Volume (subtle)
        vol = int(player.get_volume() * 100)
        cv2.putText(frame, f"Vol: {vol}%", (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imshow('3D Artistic Music Player', frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if player.is_playing:
                player.execute_action('pause')
            else:
                player.execute_action('play')
            current_action = '⏸/▶ PLAY/PAUSE'
            action_display_time = time.time()
        elif key == ord('n') or key == ord('N'):
            player.execute_action('next')
            current_action = '⏭ NEXT'
            action_display_time = time.time()
        elif key == ord('v') or key == ord('V'):
            player.execute_action('visual')
            current_action = f'🎨 {player.visual_modes[player.visual_mode]}'
            action_display_time = time.time()
        elif key == ord('+') or key == ord('='):
            player.execute_action('volume_up')
        elif key == ord('-') or key == ord('_'):
            player.execute_action('volume_down')
        elif key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
