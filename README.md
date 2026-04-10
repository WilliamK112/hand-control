# Hand Control

A hand-gesture control prototype built on top of a more modular hand-tracking foundation, focused on practical media control and future iOS-compatible adapters.

## What this version does

This version is centered on a dual-hand safety model:

- **Left hand fist** = unlock controls
- **Right hand open palm + rotation** = system volume up/down
- **Right hand index up** = Apple Music play
- **Right hand fist** = Apple Music pause
- **Right hand peace sign** = Apple Music next track

The goal is no longer just a fun gesture demo. This repo is being reshaped toward a more reusable hand-control system with clearer gesture detection, event flow, and adapter-based integrations.

## Current status

This is a **working prototype**, not the final mature release.

What is already working:
- Hand tracking with MediaPipe
- Left-hand unlock model
- Right-hand gesture detection for media actions
- macOS system volume adapter
- Apple Music adapter
- On-screen debug labels for gesture states

What still needs refinement:
- Reduce gesture false positives
- Improve left-fist reliability
- Separate gesture recognition from output adapters more cleanly
- Add a reusable event layer for future iOS control

## Demo controls

### Safety / arming
- **Left hand fist** → arm the system
- If the left hand is not recognized as a fist, right-hand actions should not trigger

### Right-hand controls
- **Open palm + rotate** → adjust system output volume
- **Index up** → Apple Music play
- **Fist** → Apple Music pause
- **Peace sign (✌️)** → Apple Music next track
  - includes an added **1.5 second interval** between repeated next-track triggers

## Run the prototype

Use a Python 3.11 virtual environment for best compatibility with MediaPipe.

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the current prototype:

```bash
python handcontrol_demo_app.py --mode handcontrol-demo --flip --hc-system-volume --hc-apple-music
```

## Demo images

Add screenshots to `demo-images/` and reference them here.

Example placeholders:
- `demo-images/handcontrol-window.png`
- `demo-images/gesture-armed.png`
- `demo-images/peace-next-track.png`

When available, include them like this:

```md
![Hand control demo](demo-images/handcontrol-window.png)
```

## Files in this repo right now

### Current prototype path
- `handcontrol_demo_app.py`
- `handcontrol_gestures.py`
- `apple_music.py`
- `system_volume.py`

### Older experimental files still present
- `gesture_dj.py`
- `gesture_music_player.py`
- `gesture_art.py`
- `animal_gestures.py`
- `artistic_3d_skeleton.py`
- `hand_tracker_new.py`

These older files are still kept for reference, but the repo is currently being shifted toward the newer hand-control prototype path above.

## Install

Minimal runtime dependencies currently listed in this repo:

```bash
pip install -r requirements.txt
```

For best compatibility, use Python 3.11.

## Why this direction

The longer-term goal is to make this project:
- more reusable
- easier to install
- less tied to one local app
- better suited for future iOS control adapters

So this version is a bridge from a fun local gesture prototype toward a more general hand-control platform.
