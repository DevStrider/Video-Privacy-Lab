"""Face tracking, clothing masks, and a lazy-loading vision model engine."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

FACE_CONFIDENCE_THRESHOLD = 0.8


def union_box(a, b):
    left, top = np.minimum(a[:2], b[:2])
    right, bottom = np.maximum(a[:2] + a[2:], b[:2] + b[2:])
    return np.array([left, top, right - left, bottom - top], dtype=np.float32)


def overlap(a, b):
    extent = np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2]))
    area = float(np.prod(extent))
    return area / max(1.0, float(np.prod(a[2:]) + np.prod(b[2:])) - area)


@dataclass
class FaceTrack:
    box: np.ndarray
    covered_box: np.ndarray
    seen_at: float
    number: int
    points: np.ndarray | None = None
    trusted_size: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    occluded: bool = False
    size_candidate: np.ndarray | None = None
    size_candidate_at: float = 0.0
    size_candidate_count: int = 0


class FaceTracker:
    """Associate detections, follow optical flow, and retain briefly hidden faces.

    Detection gaps never stop the preview. Continue optical-flow tracking and
    retain the last position if flow is unavailable, until reacquired or expired.
    hold_seconds controls the uncertainty indicator, not a blackout timeout.
    """

    def __init__(self, hold_seconds=3.0):
        if not math.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive and finite")
        self.hold_seconds = hold_seconds
        self.reset()

    def reset(self):
        self.tracks = []
        self.previous_gray = None
        self.last_time = None
        self.next_number = 1
        self.uncertain = False

    @staticmethod
    def _features(gray, box):
        """Seed only on acquisition, so a passing hand never becomes the target."""
        height, width = gray.shape
        x, y, w, h = box
        mask = np.zeros_like(gray)
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x + w)), min(height, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        mask[y0:y1, x0:x1] = 255
        return cv2.goodFeaturesToTrack(gray, 80, 0.01, 4, mask=mask)

    @staticmethod
    def _flow(previous, current, box, points):
        """Keep surviving face features; reject changed appearance and hand motion."""
        zero = np.zeros(2, dtype=np.float32)
        if points is None or len(points) < 6:
            return zero, None
        moved, status, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                     winSize=(15, 15), maxLevel=2)
        if moved is None:
            return zero, None
        returned, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None)
        if returned is None:
            return zero, None
        good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                & (np.linalg.norm(returned - points, axis=2).ravel() < 1.0)
                & (error.ravel() < 18))
        if good.sum() < 6:
            return zero, None
        deltas = (moved - points).reshape(-1, 2)[good]
        delta = np.median(deltas, axis=0)
        w, h = box[2:]
        consistent = np.linalg.norm(deltas - delta, axis=1) < max(2.0, min(w, h) * 0.06)
        surviving = moved.reshape(-1, 2)[good][consistent]
        # A cluster confined to one little visible fragment is not enough to
        # relocate the whole face. Retain the last safe position instead.
        if (consistent.mean() < 0.6 or len(surviving) < 6
                or np.any(np.ptp(surviving, axis=0) < np.array([w, h]) * 0.25)):
            return zero, None
        if np.linalg.norm(delta) > max(w, h) * 0.5:
            return zero, None
        return delta.astype(np.float32), surviving.reshape(-1, 1, 2)

    @classmethod
    def _motion(cls, previous, current, box):
        """One-shot flow for aligning delayed observations."""
        return cls._flow(previous, current, box, cls._features(previous, box))[0]

    @staticmethod
    def _confirm_size_change(track, detected, predicted, timestamp):
        """Require a stable, centered scale change before resizing a cover."""
        ratios = detected[2:] / track.trusted_size
        center_distance = np.linalg.norm(
            detected[:2] + detected[2:] / 2 - predicted[:2] - predicted[2:] / 2
        )
        uniform_scale = abs(float(ratios[0] - ratios[1])) < 0.12
        centered = center_distance < max(8.0, float(np.linalg.norm(predicted[2:])) * 0.18)
        size_change = np.any(ratios < 0.82) or np.any(ratios > 1.4)
        if not (size_change and uniform_scale and centered):
            track.size_candidate = None
            track.size_candidate_count = 0
            return False

        if (
            track.size_candidate is None
            or np.any(np.abs(detected[2:] / track.size_candidate - 1) > 0.12)
        ):
            track.size_candidate = detected[2:].copy()
            track.size_candidate_at = timestamp
            track.size_candidate_count = 1
        else:
            track.size_candidate_count += 1
        return track.size_candidate_count >= 3 and timestamp - track.size_candidate_at >= 0.2

    def update(self, gray, detections, timestamp):
        if self.last_time is not None and timestamp < self.last_time:
            raise ValueError("Frame timestamps must increase")
        if self.previous_gray is not None and self.previous_gray.shape != gray.shape:
            self.reset()
        # Abandoned tracks must not accumulate masks over the background.
        track_lifetime = max(8.0, self.hold_seconds * 3)
        self.tracks = [t for t in self.tracks if timestamp - t.seen_at <= track_lifetime]
        detections = [
            np.asarray(d, dtype=np.float32)
            for d in detections
            if len(d) == 4 and np.isfinite(d).all() and min(d[2:]) > 0
        ]
        predicted = []
        for track in self.tracks:
            box = track.box.copy()
            if self.previous_gray is not None:
                if track.points is not None:
                    delta, points = self._flow(self.previous_gray, gray, box, track.points)
                    box[:2] += delta
                    track.points = points
                    track.occluded = points is None
                elif not track.occluded:
                    box[:2] += self._motion(self.previous_gray, gray, box)
            predicted.append(box)

        # Global greedy matching, each observation and each track used once.
        candidates = []
        for ti, box in enumerate(predicted):
            for di, detection in enumerate(detections):
                iou = overlap(box, detection)
                distance = np.linalg.norm(box[:2] + box[2:] / 2 - detection[:2] - detection[2:] / 2)
                radius = max(24.0, float(np.linalg.norm(box[2:])))
                if iou > 0.08 or distance < radius:
                    candidates.append((iou - distance / (radius * 4), ti, di))
        matches, used = {}, set()
        for _, ti, di in sorted(candidates, reverse=True):
            if ti not in matches and di not in used:
                matches[ti] = di
                used.add(di)
        for ti, track in enumerate(self.tracks):
            box = predicted[ti]
            if ti in matches:
                detected = detections[matches[ti]]
                ratios = detected[2:] / track.trusted_size
                confirmed_change = self._confirm_size_change(track, detected, box, timestamp)
                partial = (np.any(detected[2:] < track.trusted_size * 0.82)
                           or np.prod(detected[2:]) < np.prod(track.trusted_size) * 0.72)
                suspicious_growth = np.any(ratios > 1.4)
                if (partial or suspicious_growth) and not confirmed_change:
                    # A partial cheek detection does not define the head's
                    # full center or trusted size.
                    track.box = box
                    track.covered_box = box.copy()
                    track.occluded = True
                else:
                    center = (0.85 * (detected[:2] + detected[2:] / 2)
                              + 0.15 * (box[:2] + box[2:] / 2))
                    # Full detections can reduce an oversized cover after a
                    # stable, proportional change in complete face detections.
                    size = detected[2:].copy() if confirmed_change else np.maximum(
                        detected[2:], track.trusted_size * 0.995)
                    track.box = np.concatenate((center - size / 2, size))
                    track.covered_box = union_box(union_box(box, track.box), detected)
                    track.trusted_size = size.copy()
                    track.seen_at = timestamp
                    track.occluded = False
                    track.points = self._features(gray, track.box)
                    if confirmed_change:
                        track.size_candidate = None
                        track.size_candidate_count = 0
            else:
                # Cover one step of motion, not the entire historical trajectory.
                # Accumulating every old box produces ever-growing rectangles.
                track.covered_box = union_box(track.box, box)
                track.box = box
        for di, detection in enumerate(detections):
            if di not in used:
                self.tracks.append(
                    FaceTrack(
                        box=detection.copy(), covered_box=detection.copy(),
                        seen_at=timestamp, number=self.next_number,
                        points=self._features(gray, detection), trusted_size=detection[2:].copy(),
                    )
                )
                self.next_number += 1
        self.uncertain = any(
            t.occluded or timestamp - t.seen_at > self.hold_seconds for t in self.tracks
        )
        boxes = []
        for track in self.tracks:
            box = track.covered_box.copy()
            age = max(0.0, timestamp - track.seen_at)
            extra = min(0.08, age * 0.03 + (0.03 if track.occluded else 0)) * box[2:]
            box[:2] -= extra
            box[2:] += 2 * extra
            boxes.append(tuple(int(round(v)) for v in box))
        self.previous_gray = gray.copy()
        self.last_time = timestamp
        return boxes


def validated_faces(rows):
    """Use complete, confident YuNet face geometry, never generic skin blobs."""
    boxes = []
    for row in (() if rows is None else rows):
        row = np.asarray(row, np.float32)
        if len(row) < 15 or not np.isfinite(row).all() or row[14] < FACE_CONFIDENCE_THRESHOLD:
            continue
        x, y, w, h = row[:4]
        if min(w, h) < 8 or not 0.35 <= w / h <= 1.35:
            continue
        landmarks = row[4:14].reshape(5, 2)
        if (np.any(landmarks < [x - w * 0.1, y - h * 0.1])
                or np.any(landmarks > [x + w * 1.1, y + h * 1.1])
                or landmarks[:2, 1].mean() >= landmarks[3:, 1].mean()):
            continue
        boxes.append(row[:4].copy())
    return boxes


class ClothingMask:
    """Motion-aligned smoothing, with current skin/background overriding history."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.gray = None
        self.alpha = None

    def update(self, gray, probabilities):
        # Official SelfieMulticlass labels: background, hair, body-skin,
        # face-skin, clothes, accessories.
        if probabilities.shape != (*gray.shape, 6):
            raise ValueError("Expected H x W x 6 segmentation confidence maps")
        clothes = probabilities[:, :, 4]
        labels = probabilities.argmax(axis=2)
        current = np.clip((clothes - 0.35) / 0.4, 0, 1).astype(np.float32)
        if self.gray is not None and self.gray.shape == gray.shape:
            scene_change = np.mean(cv2.absdiff(gray, self.gray)) > 45
            if not scene_change:
                # Backward flow maps each new pixel into the previous mask.
                flow = cv2.calcOpticalFlowFarneback(gray, self.gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                yy, xx = np.indices(gray.shape, dtype=np.float32)
                old = cv2.remap(self.alpha, xx + flow[:, :, 0], yy + flow[:, :, 1],
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                current = 0.75 * current + 0.25 * old
        # Never let smoothing paint over newly detected hands or face.
        excluded = ((labels != 4) | (probabilities[:, :, 2] > 0.25)
                    | (probabilities[:, :, 3] > 0.25))
        skin = ((probabilities[:, :, 2] > 0.25) | (probabilities[:, :, 3] > 0.25)).astype(np.uint8)
        skin = cv2.dilate(skin, np.ones((3, 3), np.uint8)) != 0
        current[excluded | skin] = 0
        current = cv2.GaussianBlur(current, (3, 3), 0)
        current[excluded | skin] = 0  # Edge softening must not undo the veto.
        self.gray, self.alpha = gray.copy(), current.copy()
        return current


class VisionEngine:
    """YuNet faces, multiclass clothing masks, and persistent temporal state."""

    def __init__(self, model_dir, hold_seconds=3.0):
        from vision_models import ensure_models

        self.faces = FaceTracker(hold_seconds)
        ensure_models(model_dir)
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is missing in this interpreter. Sync this project's dependencies."
            ) from exc
        self.mp = mp
        self.detector = cv2.FaceDetectorYN.create(
            str(model_dir / "face_detection_yunet_2023mar.onnx"), "", (320, 320),
            FACE_CONFIDENCE_THRESHOLD, 0.3, 5000,
        )
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_dir / "selfie_multiclass_256x256.tflite")
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_confidence_masks=True, output_category_mask=False,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        self.clothes = ClothingMask()
        self.timestamp_ms = -1

    def reset(self):
        self.faces.reset()
        self.clothes.reset()

    def close(self):
        self.segmenter.close()

    def analyze(self, frame, timestamp):
        """Expensive observations only, suitable for a background worker."""
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, result = self.detector.detect(frame)
        boxes = validated_faces(result)
        small = cv2.resize(frame, (256, max(32, int(height * 256 / width))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.timestamp_ms = max(self.timestamp_ms + 1, int(timestamp * 1000))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment_for_video(image, self.timestamp_ms)
        probabilities = np.stack([m.numpy_view().reshape(small.shape[:2])
                                  for m in result.confidence_masks], axis=-1)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        alpha = self.clothes.update(gray_small, probabilities)

        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        # A nearest-neighbour gate keeps resizing from painting outside the mask.
        allowed = cv2.resize(
            (self.clothes.alpha > 0).astype(np.uint8), (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha[allowed == 0] = 0
        return boxes, alpha

    def process(self, frame, timestamp):
        boxes, alpha = self.analyze(frame, timestamp)
        tracked = self.faces.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), boxes, timestamp)
        return tracked, alpha, self.faces.uncertain
