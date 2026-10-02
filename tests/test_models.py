"""Small offline integration test for the actual installed model APIs."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from vision_models import MODELS, digest
from vision_tracking import VisionEngine


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
READY = importlib.util.find_spec("mediapipe") is not None and all(
    (MODEL_DIR / name).is_file() for name in MODELS)


@unittest.skipUnless(READY, "Install dependencies and local models for integration checks")
class ModelTests(unittest.TestCase):
    def test_real_model_confidence_shapes_and_output_on_blank_frames(self):
        for name, (_, expected) in MODELS.items():
            self.assertEqual(digest(MODEL_DIR / name), expected)
        engine = VisionEngine(MODEL_DIR)
        try:
            for i, (height, width) in enumerate([(180, 320), (320, 180), (256, 256)]):
                faces, mask, lost = engine.process(np.zeros((height, width, 3), np.uint8), i / 30)
                self.assertEqual(mask.shape, (height, width))
                self.assertTrue(np.isfinite(mask).all())
                self.assertGreaterEqual(float(mask.min()), 0)
                self.assertLessEqual(float(mask.max()), 1)
                self.assertEqual(faces, [])
                self.assertFalse(lost)
        finally:
            engine.close()
