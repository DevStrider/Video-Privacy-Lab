"""Camera-rate tracking with bounded, asynchronous model inference.

The display never waits for neural inference. Measurements carry their source
image/time and are motion-aligned before being applied to newer camera frames.
"""

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from vision_tracking import FaceTracker, VisionEngine


@dataclass(frozen=True)
class Observation:
    generation: int
    timestamp: float
    frame: np.ndarray
    boxes: list
    alpha: np.ndarray


class InferenceWorker:
    """One in-flight inference, one replaceable input, one latest output."""

    def __init__(self, engine):
        self.engine = engine
        self.condition = threading.Condition()
        self.pending = None
        self.result = None
        self.error = None
        self.stopped = False
        self.generation = 0
        self.inference_fps = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True, name="vision-models")
        self.thread.start()

    def submit(self, frame, timestamp):
        with self.condition:
            self.pending = (self.generation, timestamp, frame.copy())
            self.condition.notify()

    def take(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"Vision inference failed: {self.error}") from self.error
            result, self.result = self.result, None
            return result

    def reset(self):
        with self.condition:
            self.generation += 1
            self.pending = self.result = None

    def _run(self):
        last_generation = -1
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.stopped)
                    if self.stopped:
                        return
                    generation, timestamp, frame = self.pending
                    self.pending = None
                if last_generation != generation:
                    self.engine.reset()
                    last_generation = generation
                started = time.perf_counter()
                boxes, alpha = self.engine.analyze(frame, timestamp)
                fps = 1 / max(1e-6, time.perf_counter() - started)
                with self.condition:
                    self.inference_fps = fps
                    if generation == self.generation and not self.stopped:
                        self.result = Observation(generation, timestamp, frame, boxes, alpha)
        except Exception as exc:
            with self.condition:
                self.error = exc
        finally:
            self.engine.close()

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout=5)


def warp_clothing(source, current, alpha):
    """Small optical-flow warp with a changed-pixel veto for new occluders."""
    old_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray, old_gray, None, 0.5, 2, 13, 2, 5, 1.1, 0)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    mx, my = xx + flow[:, :, 0], yy + flow[:, :, 1]
    mask = cv2.remap(alpha, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    previous_color = cv2.remap(source, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mismatch = cv2.absdiff(current, previous_color).max(axis=2) > 35
    mismatch = cv2.dilate(mismatch.astype(np.uint8), np.ones((3, 3), np.uint8))
    mask[mismatch != 0] = 0
    return mask


class RealtimeVision:
    """Track each displayed frame using the latest usable model observation."""

    def __init__(self, model_dir=None, hold_seconds=3.0, worker=None):
        self.faces = FaceTracker(hold_seconds)
        self.worker = (
            worker if worker is not None else InferenceWorker(VisionEngine(model_dir, hold_seconds))
        )
        self.previous = None
        self.alpha = None
        self.observed_at = None
        self.shape = None
        self.last_boxes = []

    @property
    def inference_fps(self):
        return self.worker.inference_fps

    def reset(self):
        self.worker.reset()
        self.faces.reset()
        self.previous = self.alpha = self.observed_at = self.shape = None
        self.last_boxes = []

    def close(self):
        self.worker.close()

    def process(self, frame, timestamp):
        height, width = frame.shape[:2]
        shape = (min(320, width), max(32, round(height * min(320, width) / width)))
        if self.shape is not None and shape != self.shape:
            self.reset()
        self.shape = shape
        small = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        motion_size = (max(32, shape[0] // 2), max(32, shape[1] // 2))
        motion_frame = cv2.resize(small, motion_size, interpolation=cv2.INTER_AREA)
        self.worker.submit(small, timestamp)
        observation = self.worker.take()
        detections = []
        if (
            observation is not None
            and observation.frame.shape == small.shape
            and 0 <= timestamp - observation.timestamp <= 0.75
        ):
            source_gray = cv2.cvtColor(observation.frame, cv2.COLOR_BGR2GRAY)
            for raw_box in observation.boxes:
                box = np.asarray(raw_box, dtype=np.float32).copy()
                # Correct delayed detections to the current image, rather than
                # snapping a mask backwards to an old neural result.
                box[:2] += FaceTracker._motion(source_gray, gray, box)
                detections.append(box)
            source = cv2.resize(observation.frame, motion_size, interpolation=cv2.INTER_AREA)
            measured = cv2.resize(observation.alpha, motion_size, interpolation=cv2.INTER_LINEAR)
            self.alpha = warp_clothing(source, motion_frame, measured)
            self.observed_at = observation.timestamp
        elif self.alpha is not None and self.previous is not None and np.any(self.alpha):
            self.alpha = warp_clothing(self.previous, motion_frame, self.alpha)
        if self.observed_at is None or timestamp - self.observed_at > 1.0:
            # Stale garment masks must not keep tinting unrelated objects.
            self.alpha = np.zeros(motion_frame.shape[:2], np.float32)
        self.previous = motion_frame
        boxes = self.faces.update(gray, detections, timestamp)
        sx, sy = width / shape[0], height / shape[1]
        boxes = [
            tuple(round(v * scale) for v, scale in zip(box, (sx, sy, sx, sy)))
            for box in boxes
        ]
        self.last_boxes = boxes
        mask = cv2.resize(self.alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        allowed = cv2.resize(
            (self.alpha > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
        mask[allowed == 0] = 0
        return boxes, mask, self.faces.uncertain
