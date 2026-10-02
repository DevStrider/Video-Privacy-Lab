#!/usr/bin/env python3
"""Experimental chaos-based frame scrambling and image-encryption metrics.

This module is deliberately separate from main.py. It is useful for studying
the lecture concepts, but the generated scramble is not a replacement for an
authenticated cryptographic cipher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def logistic_sequence(length: int, seed: float, r: float = 3.999999) -> np.ndarray:
    """Generate a deterministic logistic-map sequence in (0, 1)."""
    if not 0.0 < seed < 1.0:
        raise ValueError("seed must be strictly between 0 and 1")
    if not 3.57 < r <= 4.0:
        raise ValueError("r must be in the chaotic range (3.57, 4.0]")
    values = np.empty(length, dtype=np.float64)
    x = seed
    for index in range(length):
        x = r * x * (1.0 - x)
        values[index] = x
    return values


def _permutation(length: int, seed: float, skip: int) -> np.ndarray:
    sequence = logistic_sequence(length + skip, seed)[skip:]
    # Stable sorting makes the transform deterministic across equal values.
    return np.argsort(sequence, kind="stable")


def _inverse_permutation(permutation: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    return inverse


def arnold_cat_map(image: np.ndarray, iterations: int = 1, a: int = 1, b: int = 1) -> np.ndarray:
    """Apply the generalized Arnold map to a square image."""
    height, width = image.shape[:2]
    if height != width:
        raise ValueError("Arnold's Cat Map requires a square image")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    result = image.copy()
    size = height
    for _ in range(iterations):
        y, x = np.indices((size, size))
        new_x = (x + a * y) % size
        new_y = (b * x + (1 + a * b) * y) % size
        mapped = np.empty_like(result)
        mapped[new_y, new_x] = result[y, x]
        result = mapped
    return result


def _chaos_mask(shape: tuple[int, ...], seed: float) -> np.ndarray:
    values = logistic_sequence(int(np.prod(shape)), seed, r=3.999999)
    # Multiplication before flooring uses more than the first few decimal
    # places of the floating-point orbit while producing byte values.
    return np.floor(values * 2**53).astype(np.uint64).astype(np.uint8).reshape(shape)


def chaos_scramble(frame: np.ndarray, seed: float, inverse: bool = False) -> np.ndarray:
    """Reversible row/column permutation plus logistic-map XOR mask."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected a BGR color frame")
    height, width, _ = frame.shape
    row_perm = _permutation(height, seed, skip=32)
    col_seed = (seed * 0.6180339887498949) % 1.0
    col_perm = _permutation(width, col_seed, skip=64)
    mask = _chaos_mask(frame.shape, (seed * 0.3819660112501051) % 1.0)

    if inverse:
        unmasked = np.bitwise_xor(frame, mask)
        return unmasked[_inverse_permutation(row_perm)][:, _inverse_permutation(col_perm)]

    permuted = frame[row_perm][:, col_perm]
    return np.bitwise_xor(permuted, mask)


def entropy(image: np.ndarray) -> float:
    """Shannon entropy of an 8-bit grayscale image."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def mse(first: np.ndarray, second: np.ndarray) -> float:
    delta = first.astype(np.float64) - second.astype(np.float64)
    return float(np.mean(delta * delta))


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    value = mse(first, second)
    return float("inf") if value == 0 else float(10.0 * np.log10((255.0**2) / value))


def npcr(first: np.ndarray, second: np.ndarray) -> float:
    changed = np.any(first != second, axis=2) if first.ndim == 3 else first != second
    return float(100.0 * np.mean(changed))


def uaci(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return float(100.0 * np.mean(difference) / 255.0)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = first.astype(np.float64).ravel()
    right = second.astype(np.float64).ravel()
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def frame_metrics(plain: np.ndarray, scrambled: np.ndarray, recovered: np.ndarray) -> dict:
    return {
        "plain_entropy": entropy(plain),
        "scrambled_entropy": entropy(scrambled),
        "mse_plain_vs_scrambled": mse(plain, scrambled),
        "psnr_plain_vs_scrambled_db": psnr(plain, scrambled),
        "npcr_plain_vs_scrambled_percent": npcr(plain, scrambled),
        "uaci_plain_vs_scrambled_percent": uaci(plain, scrambled),
        "correlation_plain_vs_scrambled": correlation(plain, scrambled),
        "recovery_mse": mse(plain, recovered),
        "recovery_psnr_db": psnr(plain, recovered),
    }


def run_demo(input_path: Path, output_dir: Path, seed: float, iterations: int) -> dict:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("Input video contains no readable frames")

    scrambled = chaos_scramble(frame, seed)
    recovered = chaos_scramble(scrambled, seed, inverse=True)
    side = np.hstack((frame, scrambled, recovered))
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "plain.png"), frame)
    cv2.imwrite(str(output_dir / "scrambled.png"), scrambled)
    cv2.imwrite(str(output_dir / "recovered.png"), recovered)
    cv2.imwrite(str(output_dir / "comparison.png"), side)

    # Keep the Cat Map demonstration available for square crops of the same
    # frame, matching the lecture's equation and visual intuition.
    size = min(frame.shape[:2])
    square = frame[:size, :size]
    cat = arnold_cat_map(square, iterations=iterations)
    cv2.imwrite(str(output_dir / "arnold_cat_map.png"), cat)

    metrics = frame_metrics(frame, scrambled, recovered)
    metrics.update({"seed": seed, "arnold_iterations": iterations, "frame_shape": list(frame.shape)})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input video")
    parser.add_argument("output_dir", type=Path, help="directory for demo artifacts")
    parser.add_argument("--seed", type=float, default=0.3141592653)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    try:
        metrics = run_demo(args.input, args.output_dir, args.seed, args.iterations)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
