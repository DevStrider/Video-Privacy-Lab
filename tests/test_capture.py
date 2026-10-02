import unittest
from unittest.mock import patch

from live_capture import LatestCamera, configure_camera, open_capture


class CaptureTests(unittest.TestCase):
    def test_live_source_uses_camera_zero_and_rejects_other_indices(self):
        with patch("live_capture.cv2.VideoCapture") as capture:
            open_capture("0")
            capture.assert_called_once_with(0)
            capture.reset_mock()
            for source in ("1", "2", "00"):
                with self.subTest(source=source):
                    with self.assertRaisesRegex(ValueError, "Mac camera"):
                        open_capture(source)
            capture.assert_not_called()

    def test_saved_video_still_opens_as_a_file(self):
        with patch("live_capture.cv2.VideoCapture") as capture:
            open_capture("input.mp4")
            capture.assert_called_once_with("input.mp4")

    def test_camera_drops_backlog_and_releases_device(self):
        class FakeCamera:
            index = 0
            released = False
            def read(self):
                self.index += 1
                return self.index <= 100, self.index if self.index <= 100 else None
            def release(self):
                self.released = True
        capture = FakeCamera()
        reader = LatestCamera(capture)
        reader.thread.join(timeout=1)
        self.assertEqual(reader.read(), (True, 100))
        self.assertEqual(reader.read(), (False, None))
        reader.close()
        self.assertTrue(capture.released)
        self.assertEqual(reader.captured, 100)
        self.assertEqual(reader.dropped, 99)

    def test_device_rejecting_60fps_does_not_prevent_capture(self):
        class Camera:
            requests = []
            def set(self, prop, value):
                self.requests.append((prop, value))
                return False
        capture = Camera()
        self.assertFalse(configure_camera(capture, target_fps=60))
        self.assertEqual([v for _, v in capture.requests], [1280, 720, 60, 1])

    def test_camera_exceptions_reach_caller_and_release_device(self):
        class BrokenCamera:
            released = False
            def read(self):
                raise ValueError("disconnected")
            def release(self):
                self.released = True
        capture = BrokenCamera()
        reader = LatestCamera(capture)
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            reader.read()
        reader.close()
        self.assertTrue(capture.released)
