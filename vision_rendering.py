"""Face covers, garment styling, and preview labels.

Rendering operates on full-resolution BGR images in place. Tracking and model
inference live in separate modules so visual effects stay independent of I/O.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol

import cv2
import numpy as np

FaceBox = tuple[int, int, int, int]
OVAL_PADDING = (0.24, 0.24)
RECTANGULAR_PADDING = (0.12, 0.18)
OVAL_RADIUS_SCALE = 0.74


class FrameProcessor(Protocol):
    """Interface shared by synchronous and asynchronous vision engines."""

    def process(
        self, frame: np.ndarray, timestamp: float
    ) -> tuple[list[FaceBox], np.ndarray, bool]:
        ...


def expand_box(
    box: FaceBox, frame_shape: tuple[int, ...], x_scale: float, y_scale: float
) -> tuple[int, int, int, int]:
    """Return padded slice boundaries clipped to the image."""
    x, y, width, height = box
    frame_height, frame_width = frame_shape[:2]
    extra_x = math.ceil(width * x_scale)
    extra_y = math.ceil(height * y_scale)
    left = min(frame_width, max(0, x - extra_x))
    top = min(frame_height, max(0, y - extra_y))
    right = max(left, min(frame_width, x + width + extra_x))
    bottom = max(top, min(frame_height, y + height + extra_y))
    return left, top, right, bottom


def oval_geometry(box: FaceBox, left: int = 0, top: int = 0) -> tuple[int, int, int, int]:
    """Use the same ellipse for rendering and coverage diagnostics."""
    x, y, width, height = box
    return (
        round(x + width / 2) - left,
        round(y + height / 2) - top,
        math.ceil(width * OVAL_RADIUS_SCALE),
        math.ceil(height * OVAL_RADIUS_SCALE),
    )


def anonymize_face(frame: np.ndarray, box: FaceBox, mode: str) -> None:
    """Apply a face effect in place, leaving the surrounding image intact."""
    padding = OVAL_PADDING if mode == "oval" else RECTANGULAR_PADDING
    left, top, right, bottom = expand_box(box, frame.shape, *padding)
    region = frame[top:bottom, left:right]
    if region.size == 0:
        return

    if mode in ("oval", "shield"):
        x, y, width, height = box
        geometry = oval_geometry(box, left, top) if mode == "oval" else None
        tile, alpha, inverse = privacy_shield(*region.shape[:2], geometry)
        # At image boundaries the rounded rim can meet the face itself. Make
        # every pixel of the tracked face rectangle opaque, including clipping.
        core = (
            slice(max(0, y - top), max(0, min(bottom, y + height) - top)),
            slice(max(0, x - left), max(0, min(right, x + width) - left)),
        )
        if np.any(alpha[core] < 1):
            alpha = alpha.copy()
            alpha[core] = 1
            inverse = 1 - alpha
        region[:] = cv2.blendLinear(tile, region, alpha, inverse)
    elif mode == "solid":
        region[:] = (25, 25, 25)
    elif mode == "blur":
        kernel = max(15, (min(region.shape[:2]) // 5) | 1)
        frame[top:bottom, left:right] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    else:
        # Keep a small, fixed grid even on large close-up faces.
        tiny = cv2.resize(region, (7, 7), interpolation=cv2.INTER_AREA)
        frame[top:bottom, left:right] = cv2.resize(
            tiny, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST
        )


@lru_cache(maxsize=8)
def privacy_shield(height: int, width: int, oval_geometry=None):
    """An opaque oval/rounded cover; only its outer rim is antialiased."""
    mask = np.zeros((height, width), np.uint8)
    if oval_geometry is not None:
        cx, cy, rx, ry = oval_geometry
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1, cv2.LINE_AA)
    else:
        radius = max(2, min(height, width) // 10)
        cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
        cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
        for x in (radius, width - radius - 1):
            for y in (radius, height - radius - 1):
                cv2.circle(mask, (x, y), radius, 255, -1, cv2.LINE_AA)
    tile = np.empty((height, width, 3), np.uint8)
    ramp = np.linspace(0, 1, height)[:, None, None]
    tile[:] = np.uint8(np.array([48, 39, 30]) + ramp * np.array([20, 13, 6]))
    if oval_geometry is None:
        cx, cy = width // 2, height // 2
        unit = max(3, min(height, width) // 12)
        cv2.ellipse(
            tile, (cx, cy - unit // 2), (unit // 2, unit * 2 // 3),
            0, 180, 360, (184, 213, 207), max(1, unit // 5), cv2.LINE_AA,
        )
        cv2.rectangle(
            tile, (cx - unit, cy - unit // 4), (cx + unit, cy + unit),
            (184, 213, 207), -1, cv2.LINE_AA,
        )
        cv2.circle(tile, (cx, cy + unit // 3), max(1, unit // 5), (55, 45, 34), -1, cv2.LINE_AA)
    alpha = mask.astype(np.float32) / 255
    return tile, alpha, 1 - alpha


@lru_cache(maxsize=8)
def pattern_mask(height: int, width: int, pattern: str) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    if pattern == "grid":
        spacing = max(18, width // 18)
        mask[:, ::spacing] = 255
        mask[::spacing, :] = 255
    elif pattern == "waves":
        x = np.arange(width)
        wave = (5 * np.sin(x / 18.0)).astype(np.int32)
        for y in range(0, height, 16):
            cv2.polylines(mask, [np.column_stack((x, y + wave)).astype(np.int32)], False, 255, 1)
    return mask


def clothing_style(
    frame: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    pattern: str,
    strength: float = 0.78,
) -> None:
    """Tint segmented garment pixels while preserving their brightness."""
    if not np.any(mask):
        return
    # Restrict HD color conversions/blending to the garment's bounding region.
    active = cv2.findNonZero((mask > 0).astype(np.uint8))
    x, y, width, height = cv2.boundingRect(active)
    region = frame[y : y + height, x : x + width]
    weights = mask[y : y + height, x : x + width]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    hsv[:, :, 0] = target_hsv[0]
    hsv[:, :, 1] = np.maximum(hsv[:, :, 1], target_hsv[1])
    tinted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if pattern != "plain":
        lines = pattern_mask(*frame.shape[:2], pattern)[y : y + height, x : x + width] != 0
        tinted[lines] = (0.82 * tinted[lines] + 0.18 * 255).astype(np.uint8)

    alpha = np.clip(weights * strength, 0.0, 1.0).astype(np.float32)
    region[:] = cv2.blendLinear(tinted, region, alpha, 1 - alpha)


def draw_status(frame: np.ndarray, text: str, bottom: bool = False) -> None:
    height, width = frame.shape[:2]
    font_scale = 0.5 if width < 800 else 0.6
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    left = 16
    top = max(0, height - text_height - baseline - 32) if bottom else 16
    right = min(width, left + text_width + 24)
    end = min(height, top + text_height + baseline + 20)
    panel = frame[top:end, left:right]
    if not panel.size:
        return
    cv2.addWeighted(panel, 0.25, np.full_like(panel, (32, 27, 23)), 0.75, 0, dst=panel)
    cv2.putText(
        frame, text, (left + 12, top + text_height + 9), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (218, 233, 229), 1, cv2.LINE_AA,
    )


def coverage_fraction(shape: tuple[int, ...], boxes: list[FaceBox], mode="oval") -> float:
    """Estimate how much of the frame is covered by the face effects."""
    mask = np.zeros(shape[:2], np.uint8)
    for box in boxes:
        if mode == "oval":
            cx, cy, rx, ry = oval_geometry(box)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
        else:
            left, top, right, bottom = expand_box(box, shape, *RECTANGULAR_PADDING)
            mask[top:bottom, left:right] = 1
    return float(mask.mean())


def stylize_frame(
    frame: np.ndarray,
    engine: FrameProcessor,
    face_mode: str,
    cloth_color: tuple[int, int, int],
    pattern: str,
    timestamp: float,
    show_labels: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Process and render one frame; return face and garment presence counts."""
    faces, clothes, uncertain = engine.process(frame, timestamp)
    clothing_style(frame, clothes, cloth_color, pattern)
    for face in faces:
        anonymize_face(frame, face, face_mode)

    if show_labels:
        state = "COVER RETAINED" if uncertain else ("PRIVACY ON" if faces else "FINDING FACES")
        draw_status(frame, f"{state}  |  {len(faces)} face(s)  |  Q quit  R reset")
    return frame, len(faces), int(np.any(clothes))
