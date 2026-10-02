import threading
import unittest
from unittest.mock import patch

import numpy as np

from realtime_vision import InferenceWorker, Observation, RealtimeVision, warp_clothing


class WorkerTests(unittest.TestCase):
    def test_inference_never_blocks_submission_and_discards_results_after_reset(self):
        started, release = threading.Event(), threading.Event()
        class SlowEngine:
            closed = False
            def reset(self):
                pass
            def analyze(self, frame, timestamp):
                started.set()
                release.wait(2)
                return [], np.ones(frame.shape[:2], np.float32)
            def close(self):
                self.closed = True
        engine = SlowEngine()
        worker = InferenceWorker(engine)
        try:
            frame = np.zeros((32, 32, 3), np.uint8)
            worker.submit(frame, 0)
            self.assertTrue(started.wait(1))
            # These calls return while inference is still deliberately blocked.
            worker.submit(frame, 0.1)
            worker.submit(frame, 0.2)
            self.assertEqual(worker.pending[1], 0.2)
            self.assertIsNone(worker.take())
            worker.reset()
            release.set()
        finally:
            release.set()
            worker.close()
        self.assertIsNone(worker.take())
        self.assertTrue(engine.closed)

    def test_model_errors_are_reported_on_display_thread(self):
        class BadEngine:
            def reset(self):
                pass
            def analyze(self, *args):
                raise ValueError("bad model")
            def close(self):
                pass
        worker = InferenceWorker(BadEngine())
        worker.submit(np.zeros((32, 32, 3), np.uint8), 0)
        worker.thread.join(1)
        with self.assertRaisesRegex(RuntimeError, "bad model"):
            worker.take()
        worker.close()


class FakeWorker:
    inference_fps = 5
    result = None
    def submit(self, frame, timestamp):
        pass
    def take(self):
        result, self.result = self.result, None
        return result
    def reset(self):
        self.result = None
    def close(self):
        pass


class RealtimeTests(unittest.TestCase):
    def test_face_still_moves_between_slow_model_updates(self):
        worker = FakeWorker()
        frame = np.zeros((120, 200, 3), np.uint8)
        worker.result = Observation(0, 0, frame, [(40, 30, 50, 60)], np.zeros((120, 200), np.float32))
        engine = RealtimeVision(worker=worker)
        engine.process(frame, 0)
        with patch('realtime_vision.FaceTracker._motion', return_value=np.array([4, 0], np.float32)):
            engine.process(frame, 0.033)
            engine.process(frame, 0.066)
        self.assertEqual(round(engine.faces.tracks[0].box[0]), 48)

    def test_delayed_observation_is_aligned_before_use(self):
        worker = FakeWorker()
        frame = np.zeros((120, 200, 3), np.uint8)
        worker.result = Observation(0, 0, frame, [(40, 30, 50, 60)], np.zeros((120, 200), np.float32))
        engine = RealtimeVision(worker=worker)
        with patch('realtime_vision.FaceTracker._motion', return_value=np.array([15, 0], np.float32)):
            engine.process(frame, 0.2)
        self.assertEqual(round(engine.faces.tracks[0].box[0]), 55)

    def test_very_stale_model_results_are_not_applied(self):
        worker = FakeWorker()
        frame = np.zeros((120, 200, 3), np.uint8)
        worker.result = Observation(0, 0, frame, [(40, 30, 50, 60)], np.ones((120, 200), np.float32))
        engine = RealtimeVision(worker=worker)
        faces, alpha, _ = engine.process(frame, 2)
        self.assertEqual(faces, [])
        self.assertFalse(alpha.any())

    def test_new_occluder_clears_propagated_clothing(self):
        before = np.full((80, 100, 3), 40, np.uint8)
        after = before.copy()
        after[25:55, 30:60] = (110, 160, 220)
        alpha = warp_clothing(before, after, np.ones((80, 100), np.float32))
        self.assertTrue(np.all(alpha[30:50, 35:55] == 0))
        self.assertGreater(alpha[5, 5], 0.9)
