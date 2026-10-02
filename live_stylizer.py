#!/usr/bin/env python3
"""Live face anonymization and clothing color/style transformation.

Local face tracking and clothing segmentation with motion-aware masks.
Visual masking is not reversible encryption or guaranteed anonymization.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from live_capture import LatestCamera, configure_camera, open_capture
from realtime_vision import RealtimeVision
from vision_rendering import coverage_fraction, draw_status, stylize_frame
from vision_tracking import VisionEngine

MODEL_DIR = Path(__file__).resolve().parent / "models"
WINDOW_TITLE = "Privacy Fashion Camera"


def parse_hex_color(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("color must look like #20A4F3")
    rgb = tuple(int(match.group(1)[i : i + 2], 16) for i in (0, 2, 4))
    return rgb[2], rgb[1], rgb[0]  # OpenCV uses BGR.


def validate_options(args: argparse.Namespace) -> None:
    """Reject invalid limits before starting models, threads, or a camera."""
    if args.input.isdigit() and args.input != "0":
        raise ValueError("Live capture uses the Mac camera at index 0; use --input 0")
    positive = (
        "camera_width", "camera_height", "target_fps", "preview_width", "hold_seconds"
    )
    nonnegative = ("max_width", "benchmark_seconds", "max_frames")
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    for name in nonnegative:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative and finite")


def prepare_frame(frame: np.ndarray, max_width: int, mirror: bool) -> np.ndarray:
    """Limit processing resolution independently of the preview window size."""
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.flip(frame, 1) if mirror else frame


def open_writer(output: str, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0])
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open output video: {path}")
    return writer


def show_preview(frame: np.ndarray, width: int, initialize: bool) -> int:
    """Display one frame and return Esc when the window has been closed."""
    if initialize:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        view_width = min(width, frame.shape[1])
        view_height = round(frame.shape[0] * view_width / frame.shape[1])
        cv2.resizeWindow(WINDOW_TITLE, view_width, view_height)
    cv2.imshow(WINDOW_TITLE, frame)
    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        return 27
    return key


def report_profile(render_times: list[float], window_times: list[float]) -> None:
    render_p50, render_p95 = np.percentile(render_times, (50, 95))
    window_p50, window_p95 = np.percentile(window_times, (50, 95))
    print(
        f"Render p50/p95: {render_p50:.2f}/{render_p95:.2f} ms"
        f" | Window p50/p95: {window_p50:.2f}/{window_p95:.2f} ms",
        flush=True,
    )
    render_times.clear()
    window_times.clear()


def run(args: argparse.Namespace) -> None:
    validate_options(args)
    started = time.perf_counter()
    print("Loading face and clothing models...", flush=True)
    cv2.setNumThreads(2)
    is_camera = args.input.isdigit()
    engine_class = RealtimeVision if is_camera else VisionEngine
    engine = engine_class(MODEL_DIR, hold_seconds=args.hold_seconds)
    capture = None
    camera_reader = None
    writer = None
    preview = not args.headless and (args.preview or (is_camera and not args.output))
    processed = 0
    frame_total = 0
    active_started = None
    sample_start = started
    sample_count = 0
    display_fps = 0.0
    last_report = started
    profile_render = []
    profile_window = []
    try:
        capture = open_capture(args.input)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open {args.input}. For a camera, check System Settings > "
                "Privacy & Security > Camera for the running app, and close other camera apps."
            )
        if is_camera:
            accepted = configure_camera(
                capture, args.camera_width, args.camera_height, args.target_fps
            )
            status = "accepted" if accepted else "not supported by this camera/backend"
            print(
                f"Requested {args.camera_width}x{args.camera_height} at {args.target_fps:g} FPS; "
                f"rate request {status}.", flush=True,
            )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        if not is_camera:
            total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_total = int(total) if math.isfinite(total) else 0
        if is_camera:
            camera_reader = LatestCamera(capture)
        print(
            f"Source metadata: {fps:.1f} FPS. "
            "Actual capture, preview and model rates are measured separately.", flush=True,
        )
        print("Models ready. Q / Esc: quit. R: forget old tracks and reacquire.", flush=True)
        while True:
            if camera_reader is not None:
                ok, frame, captured_at = camera_reader.read_packet()
            else:
                ok, frame = capture.read()
                captured_at = None
            if not ok:
                break
            if active_started is None:
                active_started = time.perf_counter()
                sample_start = last_report = active_started
            frame_started = time.perf_counter()
            frame = prepare_frame(frame, args.max_width, is_camera and not args.no_mirror)
            if processed == 0:
                print(f"Preview resolution: {frame.shape[1]}x{frame.shape[0]}", flush=True)

            frame, _, _ = stylize_frame(
                frame,
                engine,
                args.face_mode,
                args.cloth_color,
                args.pattern,
                captured_at - started if camera_reader is not None else processed / fps,
                show_labels=not args.no_labels,
            )
            now = time.perf_counter()
            sample_count += 1
            if now - sample_start >= 1:
                display_fps = sample_count / (now - sample_start)
                sample_start, sample_count = now, 0
            if camera_reader is not None and now - last_report >= 5:
                print(
                    f"Capture {camera_reader.capture_fps:.1f} FPS | "
                    f"Preview {display_fps:.1f} FPS | Models {engine.inference_fps:.1f} updates/s | "
                    f"Dropped {camera_reader.dropped}",
                    flush=True,
                )
                if args.profile and profile_render:
                    report_profile(profile_render, profile_window)
                    coverage = coverage_fraction(frame.shape, engine.last_boxes, args.face_mode)
                    print(
                        f"Tracked faces: {len(engine.last_boxes)} | "
                        f"Face cover: {coverage:.1%} of view",
                        flush=True,
                    )
                last_report = now
            if camera_reader is not None and not args.no_labels:
                draw_status(
                    frame, f"Camera {camera_reader.capture_fps:.1f}  |  "
                    f"Preview {display_fps:.1f} FPS"
                    f"  |  Models {engine.inference_fps:.1f}/s", bottom=True,
                )
            if writer is None and args.output:
                writer = open_writer(args.output, frame, fps)
            if writer is not None:
                writer.write(frame)
            processed += 1

            render_finished = time.perf_counter()
            if preview:
                key = show_preview(frame, args.preview_width, initialize=processed == 1)
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    engine.reset()
            if args.profile:
                profile_render.append((render_finished - frame_started) * 1000)
                profile_window.append((time.perf_counter() - render_finished) * 1000)
            if args.max_frames and processed >= args.max_frames:
                break
            if (
                args.benchmark_seconds
                and time.perf_counter() - active_started >= args.benchmark_seconds
            ):
                break
    finally:
        # Stop the processing clock before waiting for devices/models to close.
        finished = time.perf_counter()
        if camera_reader is not None:
            camera_reader.close()
        elif capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
        engine.close()

    elapsed = time.perf_counter() - started
    active_elapsed = finished - active_started if active_started is not None else 0
    average = processed / max(active_elapsed, 1e-6)
    if args.profile and profile_render:
        report_profile(profile_render, profile_window)
    print(
        f"processed_frames={processed} active_seconds={active_elapsed:.2f} "
        f"average_fps={average:.2f} total_seconds_including_startup={elapsed:.2f}"
    )
    if frame_total:
        print(f"source_frames={frame_total}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="0",
        help="0 for the Mac camera, or an input video path",
    )
    parser.add_argument("--output", help="optional output MP4; omit for preview only")
    parser.add_argument("--preview", action="store_true", help="show a live preview window")
    parser.add_argument(
        "--preview-width", type=int, default=960,
        help="initial window width; captured image stays HD",
    )
    parser.add_argument("--profile", action="store_true", help="report render and window timing")
    parser.add_argument(
        "--face-mode", choices=("oval", "shield", "mosaic", "blur", "solid"), default="oval"
    )
    parser.add_argument("--cloth-color", type=parse_hex_color, default=parse_hex_color("#20A4F3"))
    parser.add_argument("--pattern", choices=("plain", "grid", "waves"), default="plain")
    parser.add_argument(
        "--max-width", type=int, default=1280,
        help="processing/output width limit; 0 keeps native resolution",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument(
        "--target-fps", type=float, default=30.0,
        help="requested camera rate; this Mac's built-in camera supports up to 30 FPS",
    )
    parser.add_argument("--no-mirror", action="store_true", help="disable the webcam's mirror view")
    parser.add_argument(
        "--headless", action="store_true", help="process without opening a preview window"
    )
    parser.add_argument(
        "--benchmark-seconds", type=float, default=0, help="bounded camera/processing diagnostic"
    )
    parser.add_argument(
        "--hold-seconds", type=float, default=3.0,
        help="detection age before showing the retained-cover indicator",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="stop after this many frames (0: unlimited)"
    )
    parser.add_argument("--no-labels", action="store_true", help="hide preview status panels")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("stopped")
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
