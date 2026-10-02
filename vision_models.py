"""Fetch/check the two public vision models, independently of camera startup."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "selfie_multiclass_256x256.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
    ),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_models(directory: Path) -> None:
    """Reuse verified models and atomically replace successful downloads."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODELS.items():
        target = directory / name
        if target.is_file() and digest(target) == expected:
            continue
        print(f"Downloading {name} (one time; camera frames stay local)...", flush=True)
        temporary = None
        try:
            # A partial download can never be mistaken for a ready model.
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(url, timeout=60) as response:
                    while block := response.read(1024 * 256):
                        output.write(block)
            if digest(temporary) != expected:
                raise RuntimeError(f"Checksum mismatch for {name}")
            os.replace(temporary, target)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Could not prepare {name}: {exc}. Run vision_models.py to retry."
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_models(Path(__file__).resolve().parent / "models")
    print("Face and clothing models verified.")
