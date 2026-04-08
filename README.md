# Hand Control 🎤

Real-time hand gesture recognition system using MediaPipe for controlling music playback, DJ effects, and interactive art applications.

## Features

- **Gesture DJ Controller** — Control volume, play/pause, and track switching with hand gestures
- **Music Player** — Full media control using finger gestures
- **Gesture Art** — Draw and create art with hand movements
- **Animal Gestures** — Fun animal-themed gesture recognition
- **3D Skeleton Rendering** — Visualize hand landmarks in 3D space

## Requirements

```bash
pip install opencv-python mediapipe numpy
```

Or install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the Gesture DJ Controller
python gesture_dj.py

# Run the Music Player
python gesture_music_player.py

# Run Gesture Art
python gesture_art.py
```

## Supported Gestures

| Gesture | Action |
|---------|--------|
| ✋ Open Palm | Play |
| ✊ Fist | Pause |
| 👆 Index Up | Next Track |
| 👊 Thumb Up | Volume Up |
| 👍 Thumbs Down | Volume Down |
| 🖖 Peace Sign | Toggle Effect |
| 🤘 Rock Sign | Crossfade |

## Project Structure

```
hand-control/
├── gesture_dj.py           # DJ controller with volume rotation
├── gesture_music_player.py # Music playback control
├── gesture_art.py         # Interactive art creation
├── animal_gestures.py     # Animal-themed gestures
├── artistic_3d_skeleton.py# 3D skeleton visualization
├── hand_tracker_new.py    # Base hand tracking
└── models/
    └── hand_landmarker.task # MediaPipe model
```

## Technical Details

- **Framework**: MediaPipe Hands (Google)
- **Input**: Webcam (640x480 recommended)
- **Detection**: 21 hand landmarks, 3D coordinates
- **Latency**: Real-time at 30fps

## License

MIT License — feel free to use and modify!

---

Built with ❤️ using MediaPipe
