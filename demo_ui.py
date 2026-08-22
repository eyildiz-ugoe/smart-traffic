"""Shared presentation helpers for the demo windows.

Adds the three things a live presenter needs in every case:

* an on-screen hint of the controls (they are otherwise undiscoverable),
* a pause key (freeze the frame mid-explanation),
* an end-of-run KPI card that lands the headline numbers at the moment
  attention peaks — right when the window closes.

All demo run-loops route their key handling through ``handle_display_keys``
and finish with ``show_end_card``. Actions returned to the loops:

    None      keep running
    "quit"    Q pressed — stop this demo (kiosk advances to the next)
    "exit"    ESC pressed — stop everything (kiosk exits too)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

CONTROLS_HINT = "SPACE pause   F fullscreen   Q quit"
_ESC = 27
_SPACE = 32


def draw_controls_hint(frame) -> None:
    """Small outlined control reminder in the bottom-right corner."""

    if cv2 is None:
        return
    height, width = frame.shape[:2]
    scale = max(0.42, 0.45 * width / 800)
    thickness = max(1, int(round(scale * 2.2)))
    (text_w, text_h), _ = cv2.getTextSize(
        CONTROLS_HINT, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    org = (width - text_w - 12, height - 10)
    cv2.putText(frame, CONTROLS_HINT, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, CONTROLS_HINT, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (210, 210, 210), thickness, cv2.LINE_AA)


def handle_display_keys(
    window_name: str,
    delay_ms: int,
    fullscreen_active: bool,
) -> Tuple[Optional[str], bool]:
    """Process one key poll; blocks while paused.

    Returns (action, fullscreen_active). Fullscreen toggling is applied to
    the window here so every loop behaves identically.
    """

    def _toggle(active: bool) -> bool:
        active = not active
        state = cv2.WINDOW_FULLSCREEN if active else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, state)
        return active

    key = cv2.waitKey(max(1, delay_ms)) & 0xFF
    while True:
        if key == ord("q"):
            return "quit", fullscreen_active
        if key == _ESC:
            return "exit", fullscreen_active
        if key in (ord("f"), ord("F")):
            fullscreen_active = _toggle(fullscreen_active)
        elif key == _SPACE:
            # Paused: block until space resumes (or quit/exit/fullscreen).
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == _SPACE:
                    break
                if key == ord("q"):
                    return "quit", fullscreen_active
                if key == _ESC:
                    return "exit", fullscreen_active
                if key in (ord("f"), ord("F")):
                    fullscreen_active = _toggle(fullscreen_active)
        return None, fullscreen_active


def show_end_card(
    window_name: str,
    title: str,
    lines: List[str],
    wait_seconds: float = 12.0,
    size: Tuple[int, int] = (960, 540),
) -> Optional[str]:
    """Display the closing KPI card until a key is pressed (or a timeout).

    Returns "exit" when ESC dismissed it, else None.
    """

    if cv2 is None or np is None:
        return None
    width, height = size
    card = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (width, 86), (32, 94, 26), -1)
    cv2.putText(card, title, (36, 56), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, (255, 255, 255), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(card, line, (48, 150 + index * 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(card, "any key to close", (36, height - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.imshow(window_name, card)
    waited = 0.0
    while waited < wait_seconds:
        key = cv2.waitKey(100) & 0xFF
        if key == _ESC:
            return "exit"
        if key != 255:
            return None
        waited += 0.1
    return None
