"""Check invalid limits before the live runner can open a camera or model."""

import unittest
from unittest.mock import patch

from live_stylizer import build_parser, run
from realtime_vision import RealtimeVision


class StartupValidationTests(unittest.TestCase):
    def test_invalid_limits_do_not_start_models_or_capture(self):
        cases = (
            ("--input", "1"),
            ("--target-fps", "nan"),
            ("--hold-seconds", "inf"),
            ("--hold-seconds", "0"),
            ("--benchmark-seconds", "nan"),
            ("--max-frames", "-1"),
            ("--max-width", "-1"),
            ("--camera-width", "0"),
        )
        for option, value in cases:
            with self.subTest(option=option, value=value):
                args = build_parser().parse_args([option, value])
                with patch("live_stylizer.RealtimeVision") as model:
                    with patch("live_stylizer.open_capture") as capture:
                        with self.assertRaisesRegex(
                            ValueError, "Mac camera" if option == "--input" else option
                        ):
                            run(args)
                model.assert_not_called()
                capture.assert_not_called()

    def test_invalid_hold_time_does_not_spawn_inference_worker(self):
        with patch("realtime_vision.InferenceWorker") as worker:
            with self.assertRaises(ValueError):
                RealtimeVision(hold_seconds=float("nan"))
        worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
