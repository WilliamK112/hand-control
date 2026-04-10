import argparse
import time
import cv2

from .camera import Camera
from .hands import HandDetector, landmarks_px
from .overlay import draw_hands, draw_fps, draw_label
from .gestures import count_fingers_up, is_fist_pose, is_open_palm_pose, hand_rotation_degrees


def build_argparser():
    p = argparse.ArgumentParser(description="Real-time Hand Tracker (MediaPipe + OpenCV)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--width", type=int, help="Capture width")
    p.add_argument("--height", type=int, help="Capture height")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--det", type=float, default=0.5, help="Min detection confidence")
    p.add_argument("--track", type=float, default=0.5, help="Min tracking confidence")
    p.add_argument("--flip", action="store_true", help="Mirror the camera frame")
    p.add_argument("--no-overlay", action="store_true", help="Disable drawing overlays")
    # Modes
    p.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "vmouse", "slides", "rps", "reaction", "handcontrol-demo"],
        help="Run mode: default draw, virtual mouse, slides control, rock-paper-scissors, reaction test, or hand-control demo",
    )
    # Virtual mouse options
    p.add_argument("--vm-pinch", type=float, default=0.45, help="Pinch threshold (normed 0..1) to hold click")
    p.add_argument("--vm-smooth", type=float, default=0.25, help="Pointer smoothing alpha (0..1)")
    p.add_argument("--vm-scroll", action="store_true", help="Enable scroll from pinch distance change")
    p.add_argument("--vm-scroll-gain", type=float, default=60.0, help="Scroll sensitivity")
    # Slides options
    p.add_argument("--slides-vx", type=float, default=900.0, help="Swipe velocity threshold (px/s)")
    p.add_argument("--slides-dx", type=float, default=120.0, help="Swipe distance threshold (px)")
    p.add_argument("--slides-window", type=float, default=0.25, help="Swipe time window (s)")
    p.add_argument("--slides-cooldown", type=float, default=0.8, help="Cooldown between triggers (s)")
    p.add_argument("--hc-volume-step", type=int, default=5, help="Volume step for handcontrol demo")
    p.add_argument("--hc-system-volume", action="store_true", help="Apply demo volume events to macOS system volume")
    p.add_argument("--hc-apple-music", action="store_true", help="Enable Apple Music controls in handcontrol demo")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    cam = Camera(args.camera, args.width, args.height)
    detector = HandDetector(
        max_num_hands=args.max_hands,
        model_complexity=args.complexity,
        detection_confidence=args.det,
        tracking_confidence=args.track,
    )

    # Initialize mode controllers
    vm = None
    slides = None
    game = None
    handcontrol_demo = False
    if args.mode == "vmouse":
        from .virtual_mouse import VirtualMouse
        vm = VirtualMouse(pinch_threshold=args.vm_pinch, smoothing=args.vm_smooth,
                          enable_scroll=args.vm_scroll, scroll_gain=args.vm_scroll_gain)
    elif args.mode == "slides":
        from .slides import SlideController
        slides = SlideController(vx_thresh=args.slides_vx, dx_thresh=args.slides_dx,
                                 window_sec=args.slides_window, cooldown_sec=args.slides_cooldown)
    elif args.mode == "rps":
        from .games import RPSGame
        game = RPSGame()
    elif args.mode == "reaction":
        from .games import ReactionGame
        game = ReactionGame()
    elif args.mode == "handcontrol-demo":
        handcontrol_demo = True

    volume_step_up_fn = None
    volume_step_down_fn = None
    music_play_fn = None
    music_pause_fn = None
    music_next_fn = None
    if handcontrol_demo and args.hc_system_volume:
        from .system_volume import volume_step_up, volume_step_down
        volume_step_up_fn = volume_step_up
        volume_step_down_fn = volume_step_down
    if handcontrol_demo and args.hc_apple_music:
        from .apple_music import play, pause, next_track
        music_play_fn = play
        music_pause_fn = pause
        music_next_fn = next_track

    last_volume_dir = None
    last_volume_ts = 0.0
    last_rotation_deg = None
    last_play_ts = 0.0
    last_pause_ts = 0.0
    last_next_ts = 0.0
    rotation_delta_threshold_deg = 8.0
    cooldown_sec = 0.12
    discrete_cooldown_sec = 0.60
    peace_cooldown_sec = 1.50
    required_hold_frames = 3
    gesture_hold_counts = {
        "right_fist": 0,
        "right_index_up": 0,
        "right_open_palm": 0,
        "right_peace": 0,
    }

    prev_t = time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                # Warm-up retry: some backends return False on the first read
                for _ in range(10):
                    ok, frame = cam.read()
                    if ok:
                        break
                    cv2.waitKey(1)
                    time.sleep(0.05)
                if not ok:
                    print("Failed to read from camera; retrying...")
                    continue
            if args.flip:
                frame = cv2.flip(frame, 1)
            results = detector.process(frame)

            hand_infos = []
            if getattr(results, "multi_hand_landmarks", None):
                handedness_list = getattr(results, "multi_handedness", [])
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, handedness_list):
                    label = (
                        handedness.classification[0].label
                        if handedness and getattr(handedness, "classification", None)
                        else "Hand"
                    )
                    count, finger_states = count_fingers_up(frame, hand_landmarks, label)
                    pts = landmarks_px(frame, hand_landmarks)
                    rotation_deg = hand_rotation_degrees(frame, hand_landmarks)
                    hand_infos.append(
                        {
                            "label": label,
                            "landmarks": hand_landmarks,
                            "count": count,
                            "finger_states": finger_states,
                            "pts": pts,
                            "is_fist": is_fist_pose(frame, hand_landmarks, label),
                            "is_open_palm": is_open_palm_pose(frame, hand_landmarks, label),
                            "rotation_deg": rotation_deg,
                        }
                    )

            if not args.no_overlay:
                draw_hands(frame, results, draw=True)
                for info in hand_infos:
                    pts = info["pts"]
                    if pts:
                        x, y = pts[0]
                        draw_label(frame, f"{info['label']}: {info['count']}", (x, max(20, y - 10)))

            armed = any(info["label"].lower().startswith("left") and info["is_fist"] for info in hand_infos)
            right_open_palm = next(
                (info for info in hand_infos if info["label"].lower().startswith("right") and info["is_open_palm"]),
                None,
            )

            volume_event = None
            music_event = None
            right_index_up = next(
                (
                    info for info in hand_infos
                    if info["label"].lower().startswith("right")
                    and info["finger_states"]["Index"]
                    and not info["finger_states"]["Middle"]
                    and not info["finger_states"]["Ring"]
                    and not info["finger_states"]["Pinky"]
                ),
                None,
            )
            right_peace = next(
                (
                    info for info in hand_infos
                    if info["label"].lower().startswith("right")
                    and info["finger_states"]["Index"]
                    and info["finger_states"]["Middle"]
                    and not info["finger_states"]["Ring"]
                    and not info["finger_states"]["Pinky"]
                ),
                None,
            )
            right_fist = next(
                (info for info in hand_infos if info["label"].lower().startswith("right") and info["is_fist"]),
                None,
            )

            gesture_hold_counts["right_fist"] = gesture_hold_counts["right_fist"] + 1 if right_fist is not None else 0
            gesture_hold_counts["right_index_up"] = gesture_hold_counts["right_index_up"] + 1 if right_index_up is not None else 0
            gesture_hold_counts["right_open_palm"] = gesture_hold_counts["right_open_palm"] + 1 if right_open_palm is not None else 0
            gesture_hold_counts["right_peace"] = gesture_hold_counts["right_peace"] + 1 if right_peace is not None else 0
            if handcontrol_demo and armed and right_open_palm is not None:
                rotation = right_open_palm["rotation_deg"]
                now_ts = time.time()
                if last_rotation_deg is not None:
                    rotation_delta = rotation - last_rotation_deg
                    if rotation_delta >= rotation_delta_threshold_deg and (last_volume_dir != "up" or now_ts - last_volume_ts >= cooldown_sec):
                        volume_event = "VOLUME_UP"
                        last_volume_dir = "up"
                        last_volume_ts = now_ts
                    elif rotation_delta <= -rotation_delta_threshold_deg and (last_volume_dir != "down" or now_ts - last_volume_ts >= cooldown_sec):
                        volume_event = "VOLUME_DOWN"
                        last_volume_dir = "down"
                        last_volume_ts = now_ts
                last_rotation_deg = rotation
            elif handcontrol_demo:
                last_volume_dir = None
                last_rotation_deg = None

            if handcontrol_demo and armed:
                now_ts = time.time()
                if right_fist is not None and music_pause_fn is not None and gesture_hold_counts["right_fist"] >= required_hold_frames and now_ts - last_pause_ts >= discrete_cooldown_sec:
                    music_event = "PAUSE"
                    last_pause_ts = now_ts
                    music_pause_fn()
                    gesture_hold_counts["right_fist"] = 0
                elif right_peace is not None and music_next_fn is not None and gesture_hold_counts["right_peace"] >= required_hold_frames and now_ts - last_next_ts >= peace_cooldown_sec:
                    music_event = "NEXT_TRACK"
                    last_next_ts = now_ts
                    music_next_fn()
                    gesture_hold_counts["right_peace"] = 0
                elif right_index_up is not None and music_play_fn is not None and gesture_hold_counts["right_index_up"] >= required_hold_frames and now_ts - last_play_ts >= discrete_cooldown_sec:
                    music_event = "PLAY"
                    last_play_ts = now_ts
                    music_play_fn()
                    gesture_hold_counts["right_index_up"] = 0

            if handcontrol_demo:
                left_fist_detected = any(info["label"].lower().startswith("left") and info["is_fist"] for info in hand_infos)
                draw_label(frame, f"STATE: {'ARMED' if armed else 'SAFE'}", (20, 30))
                draw_label(frame, f"LEFT FIST: {'YES' if left_fist_detected else 'NO'}", (20, 60))
                draw_label(frame, f"RIGHT FIST: {'YES' if right_fist is not None else 'NO'}", (20, 90))
                draw_label(frame, f"RIGHT INDEX: {'YES' if right_index_up is not None else 'NO'}", (20, 120))
                draw_label(frame, f"RIGHT PEACE: {'YES' if right_peace is not None else 'NO'}", (20, 150))
                draw_label(frame, f"RIGHT PALM: {'YES' if right_open_palm is not None else 'NO'}", (20, 180))
                if right_open_palm is not None:
                    draw_label(frame, f"RIGHT ROT: {right_open_palm['rotation_deg']:.1f} deg", (20, 210))
                else:
                    draw_label(frame, "RIGHT ROT: n/a", (20, 210))
                draw_label(frame, f"EVENT: {volume_event or music_event or '-'}", (20, 240))
                draw_label(frame, f"SYSVOL: {'ON' if args.hc_system_volume else 'OFF'}", (20, 270))
                draw_label(frame, f"MUSIC: {'ON' if args.hc_apple_music else 'OFF'}", (20, 300))
                draw_label(frame, f"HOLDS F/I/P: {gesture_hold_counts['right_fist']}/{gesture_hold_counts['right_index_up']}/{gesture_hold_counts['right_peace']}", (20, 330))
                if volume_event:
                    print(volume_event, flush=True)
                    if volume_event == "VOLUME_UP" and volume_step_up_fn is not None:
                        volume_step_up_fn(args.hc_volume_step)
                    elif volume_event == "VOLUME_DOWN" and volume_step_down_fn is not None:
                        volume_step_down_fn(args.hc_volume_step)
                if music_event:
                    print(music_event, flush=True)

            # Mode-specific updates
            if args.mode == "vmouse" and vm is not None:
                vm.update(frame, results)
            elif args.mode == "slides" and slides is not None:
                slides.update(frame, results)
            elif args.mode in ("rps", "reaction") and game is not None:
                game.update(frame, results)

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_t)
            prev_t = now
            draw_fps(frame, fps)

            cv2.imshow("Hand Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("h"):
                args.no_overlay = not args.no_overlay

    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

