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

from ui_text import T

CONTROLS_HINT = "SPACE pause   F fullscreen   Q quit"
_ESC = 27
_SPACE = 32

def tint_zone(frame, rect, color, alpha: float = 0.25) -> None:
    """Tint the road area of a detection zone directly in the zone's colour.

    Real-mode standard: monitored regions read as coloured pavement, not as
    boxes drawn over the video. Blends in place on the zone ROI only.
    """

    if np is None:
        return
    x, y, w, h = rect
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(fw, int(x + w)), min(fh, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

# ---------------------------------------------------------------------------
# Unicode-capable text rendering.
#
# OpenCV's Hershey fonts are ASCII-only — Turkish characters (Ş, ğ, İ, …)
# would render as '?'. All demo text therefore goes through a PIL renderer.
# Each unique (text, size, color) is rasterized ONCE into an RGBA sprite
# and cached; per-frame drawing is a fast numpy alpha-blend.
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow ships with ultralytics
    Image = None  # type: ignore[assignment]

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
_FONTS: dict = {}
_SPRITES: dict = {}


def _font(size: int):
    font = _FONTS.get(size)
    if font is None:
        for path in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:  # pragma: no cover - no TTF available
            font = ImageFont.load_default()
        _FONTS[size] = font
    return font


def _sprite(text: str, size: int, color, outline: bool):
    key = (text, size, color, outline)
    cached = _SPRITES.get(key)
    if cached is not None:
        return cached
    font = _font(size)
    stroke = max(1, size // 9) if outline else 0
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    width, height = max(1, right - left), max(1, bottom - top)
    image = Image.new("RGBA", (width + 2, height + 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    rgb = (color[2], color[1], color[0])  # BGR -> RGB
    draw.text((-left + 1, -top + 1), text, font=font, fill=rgb + (255,),
              stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    rgba = np.asarray(image, dtype=np.uint8)
    sprite = (rgba[:, :, 2::-1].astype(np.float32),        # back to BGR
              (rgba[:, :, 3:4].astype(np.float32)) / 255.0)
    if len(_SPRITES) > 512:
        _SPRITES.clear()
    _SPRITES[key] = sprite
    return sprite


def text_size(text: str, size: int = 16) -> Tuple[int, int]:
    """Pixel (width, height) the text will occupy."""

    rgb, _ = _sprite(text, size, (255, 255, 255), True)
    return rgb.shape[1], rgb.shape[0]


def draw_text(frame, text: str, org: Tuple[int, int], size: int = 16,
              color=(255, 255, 255), outline: bool = True,
              center: bool = False) -> None:
    """Draw unicode text (top-left anchored, or centred on org)."""

    if not text or Image is None:
        return
    rgb, alpha = _sprite(text, size, tuple(color), outline)
    h, w = rgb.shape[:2]
    x, y = (int(org[0] - w // 2), int(org[1] - h // 2)) if center else (int(org[0]), int(org[1]))
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    sub_rgb = rgb[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    sub_a = alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    frame[y0:y1, x0:x1] = (sub_rgb * sub_a + roi * (1.0 - sub_a)).astype(np.uint8)


def draw_controls_hint(frame) -> None:
    """Small outlined control reminder in the bottom-right corner."""

    if cv2 is None:
        return
    height, width = frame.shape[:2]
    text = T(CONTROLS_HINT)
    size = max(12, int(14 * width / 900))
    w, h = text_size(text, size)
    draw_text(frame, text, (width - w - 10, height - h - 6), size=size,
              color=(210, 210, 210))


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
    draw_text(card, title, (36, 28), size=34, color=(255, 255, 255))
    for index, line in enumerate(lines):
        draw_text(card, line, (48, 128 + index * 44), size=22, color=(230, 230, 230))
    draw_text(card, T("any key to close"), (36, height - 40), size=15,
              color=(140, 140, 140), outline=False)
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
