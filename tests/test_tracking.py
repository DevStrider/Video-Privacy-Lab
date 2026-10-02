"""Regression tests for movement, detection gaps, and hand occlusion."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vision_rendering import anonymize_face, clothing_style, stylize_frame
from vision_tracking import ClothingMask, FaceTracker, validated_faces


class FaceTrackingTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.zeros((120, 200), np.uint8)
        self.tracker = FaceTracker(hold_seconds=1)

    def test_brief_occlusion_keeps_mask_then_reacquires_same_track(self):
        self.tracker.update(self.gray, [(40, 30, 50, 60)], 0)
        with patch.object(FaceTracker, "_motion", return_value=np.array([8, 0], np.float32)):
            boxes = self.tracker.update(self.gray, [], 0.3)
        self.assertEqual(len(boxes), 1)
        self.assertFalse(self.tracker.uncertain)
        x, y, w, h = boxes[0]
        self.assertLessEqual(x, 40)
        self.assertGreaterEqual(x + w, 98)
        self.tracker.update(self.gray, [(50, 30, 50, 60)], 0.6)
        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertEqual(self.tracker.tracks[0].number, 1)

    def test_long_loss_keeps_local_mask_and_live_background(self):
        self.tracker.update(self.gray, [(40, 30, 50, 60)], 0)
        self.tracker.update(self.gray, [], 1.1)
        self.assertTrue(self.tracker.uncertain)
        class LostEngine:
            def process(self, frame, timestamp):
                return [(40, 30, 50, 60)], np.zeros(frame.shape[:2]), True
        frame = np.full((240, 320, 3), 150, np.uint8)
        output, _, _ = stylize_frame(frame, LostEngine(), "mosaic", (255, 0, 0), "plain", 1.1, False)
        self.assertTrue(np.all(output[150:] == 150))
        self.tracker.update(self.gray, [(42, 30, 50, 60)], 1.2)
        self.assertFalse(self.tracker.uncertain)

    def test_tracking_continues_past_detection_gap_without_growing_trail(self):
        self.tracker.update(self.gray, [(40, 30, 50, 60)], 0)
        with patch.object(FaceTracker, "_motion", return_value=np.array([8, 0], np.float32)):
            for i in range(1, 10):
                boxes = self.tracker.update(self.gray, [], i * 0.2)
        self.assertEqual(round(self.tracker.tracks[0].box[0]), 112)
        self.assertLess(boxes[0][2], 100)

    def test_occlusion_does_not_shrink_mask_to_visible_face_fragment(self):
        self.tracker.update(self.gray, [(40, 30, 50, 60)], 0)
        boxes = self.tracker.update(self.gray, [(60, 35, 20, 40)], 0.1)
        self.assertGreaterEqual(boxes[0][2], 49)
        self.assertGreaterEqual(boxes[0][3], 58)

    def test_repeated_cheek_detections_do_not_shift_or_erode_cover(self):
        self.tracker.update(self.gray, [(40, 30, 50, 60)], 0)
        for i in range(1, 120):
            self.tracker.update(self.gray, [(70, 35, 18, 40)], i / 60)
        np.testing.assert_allclose(self.tracker.tracks[0].box, [40, 30, 50, 60])
        self.assertTrue(self.tracker.uncertain)
        self.tracker.update(self.gray, [(42, 30, 50, 60)], 2.1)
        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertFalse(self.tracker.uncertain)

    def test_foreground_occluder_does_not_become_tracking_target(self):
        rng = np.random.default_rng(11)
        gray = np.full((120, 200), 80, np.uint8)
        gray[25:95, 45:115] = rng.integers(0, 256, (70, 70), dtype=np.uint8)
        tracker = FaceTracker()
        tracker.update(gray, [(45, 25, 70, 70)], 0)
        occluded = gray.copy()
        occluded[20:100, 40:120] = 180
        tracker.update(occluded, [], 1 / 60)
        # After complete occlusion, move a textured "hand" across the face.
        hand = rng.integers(0, 256, (80, 80), dtype=np.uint8)
        for i in range(1, 12):
            frame = occluded.copy()
            frame[20:100, 40 + i:120 + i] = hand
            tracker.update(frame, [], (i + 1) / 60)
        np.testing.assert_allclose(tracker.tracks[0].box, [45, 25, 70, 70], atol=1)
        self.assertTrue(tracker.uncertain)

    def test_partial_occlusion_still_follows_visible_face_features(self):
        rng = np.random.default_rng(22)
        gray = np.full((160, 240), 80, np.uint8)
        gray[40:120, 60:140] = rng.integers(0, 256, (80, 80), dtype=np.uint8)
        tracker = FaceTracker()
        tracker.update(gray, [(60, 40, 80, 80)], 0)
        shifted = cv2.warpAffine(gray, np.float32([[1, 0, 4], [0, 1, 2]]), (240, 160))
        shifted[65:125, 95:145] = 180
        tracker.update(shifted, [], 1 / 60)
        np.testing.assert_allclose(tracker.tracks[0].box[:2], [64, 42], atol=1)

    def test_giant_false_observation_cannot_expand_an_existing_cover(self):
        self.tracker.update(self.gray, [(40, 30, 50, 60)], 0)
        self.tracker.update(self.gray, [(0, 0, 200, 120)], 0.2)
        np.testing.assert_allclose(self.tracker.tracks[0].box, [40, 30, 50, 60])
        self.assertLess(self.tracker.tracks[0].covered_box[2], 100)

    def test_confident_scale_change_can_reduce_a_previously_large_mask(self):
        rng = np.random.default_rng(42)
        gray = np.full((160, 240), 80, np.uint8)
        texture = rng.integers(0, 256, (80, 80), dtype=np.uint8)
        gray[40:120, 60:140] = cv2.GaussianBlur(texture, (5, 5), 0)
        tracker = FaceTracker()
        tracker.update(gray, [(60, 40, 80, 80)], 0)
        smaller = cv2.warpAffine(gray, np.float32([[0.8, 0, 20], [0, 0.8, 16]]), (240, 160))
        for timestamp in (0.1, 0.25, 0.4, 0.55):
            tracker.update(smaller, [(68, 48, 64, 64)], timestamp)
        self.assertLessEqual(float(tracker.tracks[0].box[2]), 65)
        self.assertFalse(tracker.uncertain)

    def test_abandoned_track_expires_instead_of_piling_up_background_masks(self):
        self.tracker.update(self.gray, [(40, 30, 50, 60)], 0)
        self.assertEqual(self.tracker.update(self.gray, [], 9), [])

    def test_low_confidence_or_invalid_face_geometry_is_not_a_new_face(self):
        good = np.array([40, 30, 50, 60, 52, 45, 77, 45, 65, 60, 54, 78, 76, 78, 0.95], np.float32)
        weak = good.copy()
        weak[14] = 0.7
        invalid = good.copy()
        invalid[4:14] = [0, 0] * 5
        self.assertEqual(len(validated_faces([good, weak, invalid])), 1)

    def test_shield_core_removes_all_original_detail_even_at_screen_edge(self):
        rng = np.random.default_rng(8)
        for box in [(40, 30, 50, 60), (-10, -10, 50, 60), (175, 100, 50, 60)]:
            first = rng.integers(0, 256, (120, 200, 3), dtype=np.uint8)
            second = 255 - first
            background = first[110, 5].copy()
            anonymize_face(first, box, "shield")
            anonymize_face(second, box, "shield")
            x, y, w, h = box
            core = np.s_[max(0, y):min(120, y + h), max(0, x):min(200, x + w)]
            np.testing.assert_array_equal(first[core], second[core])
            np.testing.assert_array_equal(first[110, 5], background)

    def test_oval_keeps_corners_clear_and_face_opaque_including_clipping(self):
        rng = np.random.default_rng(10)
        for box in [(60, 40, 80, 80), (-10, -10, 50, 60), (175, 100, 50, 60)]:
            first = rng.integers(0, 256, (160, 240, 3), dtype=np.uint8)
            second = 255 - first
            corner = first[21, 41].copy()
            anonymize_face(first, box, "oval")
            anonymize_face(second, box, "oval")
            x, y, w, h = box
            core = np.s_[max(0, y):min(160, y + h), max(0, x):min(240, x + w)]
            np.testing.assert_array_equal(first[core], second[core])
            if box == (60, 40, 80, 80):
                np.testing.assert_array_equal(first[21, 41], corner)

    def test_fast_detection_is_covered_without_smoothing_lag(self):
        self.tracker.update(self.gray, [(40, 30, 50, 60)], 0)
        x, _, w, _ = self.tracker.update(self.gray, [(70, 30, 50, 60)], 0.1)[0]
        self.assertGreaterEqual(x + w, 120)

    def test_multiple_faces_match_independently(self):
        self.tracker.update(self.gray, [(10, 10, 30, 40), (130, 20, 30, 40)], 0)
        self.tracker.update(self.gray, [(135, 20, 30, 40), (15, 10, 30, 40)], 0.1)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertLess(self.tracker.tracks[0].box[0], 50)
        self.assertGreater(self.tracker.tracks[1].box[0], 100)

    def test_real_optical_flow_follows_translated_texture(self):
        rng = np.random.default_rng(9)
        gray = rng.integers(0, 256, (120, 200), dtype=np.uint8)
        shifted = cv2.warpAffine(gray, np.float32([[1, 0, 5], [0, 1, 2]]), (200, 120))
        shift = FaceTracker._motion(gray, shifted, np.array([40, 30, 60, 60]))
        np.testing.assert_allclose(shift, [5, 2], atol=0.6)

    def test_offscreen_box_does_not_wrap_negative_array_slice(self):
        frame = np.full((40, 40, 3), 180, np.uint8)
        anonymize_face(frame, (-100, -100, 10, 10), "solid")
        self.assertTrue(np.all(frame == 180))


class ClothingTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((48, 64), 100, np.uint8)
        self.probs = np.zeros((48, 64, 6), np.float32)
        self.probs[:, :, 4] = 1
        self.mask = ClothingMask()

    def test_hand_immediately_clears_old_clothing_mask_and_pixels(self):
        self.mask.update(self.gray, self.probs)
        self.probs[15:35, 20:40] = 0
        self.probs[15:35, 20:40, 2] = 1
        alpha = self.mask.update(self.gray, self.probs)
        self.assertTrue(np.all(alpha[15:35, 20:40] == 0))
        self.assertGreater(alpha[5, 5], 0.9)
        frame = np.full((48, 64, 3), 120, np.uint8)
        clothing_style(frame, alpha, (255, 0, 0), "plain")
        self.assertTrue(np.all(frame[15:35, 20:40] == 120))
        self.assertFalse(np.all(frame[5, 5] == 120))
        self.probs[15:35, 20:40] = 0
        self.probs[15:35, 20:40, 4] = 1
        returned = self.mask.update(self.gray, self.probs)
        self.assertGreater(returned[25, 30], 0.7)

    def test_background_does_not_retain_clothing_ghost(self):
        self.mask.update(self.gray, self.probs)
        self.probs[:] = 0
        self.probs[:, :, 0] = 1
        self.assertFalse(np.any(self.mask.update(self.gray, self.probs)))

    def test_smoothing_reduces_small_confidence_flicker(self):
        self.mask.update(self.gray, self.probs)
        self.probs[:, :, 4] = 0.6
        self.probs[:, :, 0] = 0.4
        alpha = self.mask.update(self.gray, self.probs)
        self.assertGreater(alpha[20, 20], (0.6 - 0.35) / 0.4)

    def test_reset_clears_history(self):
        self.mask.update(self.gray, self.probs)
        self.mask.reset()
        self.assertIsNone(self.mask.alpha)


if __name__ == "__main__":
    unittest.main()
