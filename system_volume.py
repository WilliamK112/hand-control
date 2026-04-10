import subprocess


def _run_osascript(script: str) -> None:
    subprocess.run(["osascript", "-e", script], check=False)


def volume_step_up(step: int = 5) -> None:
    _run_osascript(f'set volume output volume ((output volume of (get volume settings)) + {step})')


def volume_step_down(step: int = 5) -> None:
    _run_osascript(f'set volume output volume ((output volume of (get volume settings)) - {step})')
