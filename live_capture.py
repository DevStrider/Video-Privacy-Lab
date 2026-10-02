"""Keep webcam latency bounded by retaining only the latest unread frame."""

import threading
import time

import cv2


def open_capture(source: str):
    """Open the Mac camera at index 0 or a saved video file."""
    if source.isdigit():
        if source != "0":
            raise ValueError("Live capture uses the Mac camera at index 0; use --input 0")
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(source)


def configure_camera(capture, width=1280, height=720, target_fps=30.0):
    """Request a capture mode; drivers may negotiate different settings."""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    accepted = capture.set(cv2.CAP_PROP_FPS, target_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return accepted


class LatestCamera:
    """Read in the background and replace unread frames instead of queuing."""

    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.frame = None
        self.timestamp = None
        self.capture_fps = 0.0
        self.captured = 0
        self.dropped = 0
        self.done = False
        self.error = None
        self.thread = threading.Thread(target=self._pump, daemon=True, name="camera-reader")
        self.thread.start()

    def _pump(self):
        sample_start = None
        sample_count = 0
        try:
            while not self.stopping.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    break
                now = time.perf_counter()
                with self.condition:
                    self.captured += 1
                    if sample_start is None:
                        sample_start = now
                    else:
                        sample_count += 1
                    if now - sample_start >= 1:
                        self.capture_fps = sample_count / (now - sample_start)
                        sample_start, sample_count = now, 0
                    if self.frame is not None:
                        self.dropped += 1
                    self.frame = frame
                    self.timestamp = now
                    self.condition.notify_all()
        except Exception as exc:
            self.error = exc
        finally:
            self.capture.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read(self, timeout=10):
        ok, frame, _ = self.read_packet(timeout)
        return ok, frame

    def read_packet(self, timeout=10):
        with self.condition:
            if not self.condition.wait_for(lambda: self.frame is not None or self.done, timeout):
                raise RuntimeError(
                    "Camera opened but no frames arrived. Check the running app's Camera permission."
                )
            if self.error is not None:
                raise RuntimeError(f"Camera read failed: {self.error}") from self.error
            frame, self.frame = self.frame, None
            return frame is not None, frame, self.timestamp

    def close(self):
        self.stopping.set()
        # Only the capture thread releases the device; never race read/release.
        self.thread.join(timeout=2)
