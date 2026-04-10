import subprocess


def _run(script: str) -> None:
    subprocess.run(["osascript", "-e", script], check=False)


def play() -> None:
    _run('tell application "Music" to play')


def pause() -> None:
    _run('tell application "Music" to pause')


def next_track() -> None:
    _run('tell application "Music" to next track')


def previous_track() -> None:
    _run('tell application "Music" to previous track')


def player_state() -> str:
    out = subprocess.run(
        ["osascript", "-e", 'tell application "Music" to get player state as string'],
        capture_output=True,
        text=True,
        check=False,
    )
    return (out.stdout or "").strip()
