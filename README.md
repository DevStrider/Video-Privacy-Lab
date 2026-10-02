# Video Encryption & Live Privacy Camera

A local multimedia-security project with three runnable tools:

| Tool | Purpose |
| --- | --- |
| `live_stylizer.py` | Track and cover faces, recolor clothing, and show a smooth camera preview. |
| `main.py` | Encrypt decoded video frames into a password-protected `.venc` container using AES-256-GCM. |
| `chaos_lab.py` | Explore reversible chaotic scrambling and image metrics for coursework. |

The camera effect is visual masking. Password-based video encryption is a separate workflow in `main.py`.

## Setup

Use Python **3.13 or newer**. The current environment was tested with Python 3.13.5 on macOS.

From the project directory, install the locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

All examples below use the project interpreter, so activating the environment is optional. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Use `opencv-contrib-python`; installing another OpenCV package alongside it can cause conflicts because both provide `cv2`.

Prepare the vision models once:

```bash
.venv/bin/python vision_models.py
```

Camera startup also checks the models and downloads missing or invalid files. Downloads are verified against SHA-256 hashes before use. Internet access is needed to fetch models; camera frames and inference stay local.

For PyCharm, select `.venv/bin/python` as the project interpreter, use this folder as the working directory, and run `live_stylizer.py` without parameters. On macOS, grant Camera access to the app running Python under **System Settings → Privacy & Security → Camera**.

## Live camera

```bash
.venv/bin/python live_stylizer.py
```

Live input uses camera index **0** for the Mac camera, requests **1280 × 720 at 30 FPS**, and opens a mirrored preview with a smooth, opaque **oval** over each tracked face. Clothing receives a plain blue tint. Other camera indices are not selectable.

The preview window initially opens at 960 pixels wide. That window size does not reduce the underlying HD frame. `--max-width` controls the processing/output resolution independently.

| Control | Action |
| --- | --- |
| **Q** or **Esc** | Quit and release the camera. |
| **R** | Clear face tracks and clothing history, then reacquire the scene. |
| Close the preview window | Quit. |

Change the appearance:

```bash
.venv/bin/python live_stylizer.py --cloth-color '#20A4F3' --pattern waves
.venv/bin/python live_stylizer.py --face-mode solid --no-labels
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | `0` | `0` for the Mac camera, or a saved video file path. |
| `--face-mode` | `oval` | `oval`, rounded `shield`, `mosaic`, `blur`, or `solid`. |
| `--cloth-color` | `#20A4F3` | Garment tint as an RGB hex color. |
| `--pattern` | `plain` | `plain`, `grid`, or `waves`. |
| `--camera-width`, `--camera-height` | `1280`, `720` | Requested capture dimensions. |
| `--target-fps` | `30` | Requested capture rate; delivered rate depends on the device/backend. |
| `--max-width` | `1280` | Frame width limit; `0` retains native input resolution. |
| `--preview-width` | `960` | Initial preview window width. |
| `--hold-seconds` | `3` | Detection age before the cover is marked as retained. |
| `--no-mirror` | Off | Disable webcam mirroring. Files are never mirrored automatically. |
| `--no-labels` | Off | Hide status panels. |
| `--output` | None | Write the processed frames to an MP4. |
| `--preview` | Off | Explicitly show a preview, including while recording or processing a file. |
| `--headless` | Off | Suppress the preview window. |
| `--profile` | Off | Report render/window timing and, for cameras, estimated face coverage. |
| `--benchmark-seconds` | `0` | Stop after this processing duration; `0` disables the limit. |
| `--max-frames` | `0` | Stop after this many frames; `0` disables the limit. |

Without `--output`, webcam input opens the preview automatically. Use `--preview` to display it while saving a recording:

```bash
.venv/bin/python live_stylizer.py --output tmp/camera.mp4 --preview
```

### Tracking and preview performance

The camera reader keeps only the latest unread frame. A background worker performs face detection and clothing segmentation with one replaceable pending frame, avoiding a backlog of old images. Optical flow updates face positions and clothing masks between model results; delayed detections are aligned to the current frame, and stale results are discarded.

New face tracks require confident YuNet detections and plausible landmarks. Clothing or skin segmentation does not create face tracks. During brief hand occlusion, surviving face features can keep the cover moving. If motion becomes unreliable, the cover stays at the estimated position until a reliable detection returns. Partial detections do not immediately shrink the trusted face size, and large size changes require repeated confirmation.

The oval covers the tracked face rectangle with an opaque interior and a softened outer rim. Uncertainty padding is bounded, and covers include one step of motion rather than the entire movement history. Tracks expire after `max(8, 3 × hold_seconds)` seconds without a reliable detection: **9 seconds by default**.

Clothing masks come from multiclass segmentation. Motion smoothing and skin/background checks reduce flicker and recoloring of hands. Appearance changes clear propagated garment masks, although fast motion can temporarily leave gaps in the tint.

Measure the actual rates:

```bash
# Visible Mac camera preview and timings
.venv/bin/python live_stylizer.py --profile --benchmark-seconds 20

# Processing diagnostic without window overhead
.venv/bin/python live_stylizer.py --headless --profile --benchmark-seconds 20
```

The console reports **capture FPS**, **preview/processing FPS**, **model updates per second**, and **dropped camera frames** separately. Headless “Preview” measures processed frames, not visible display refresh. Neural models can update more slowly while the preview continues at the camera rate. The final average excludes model/camera shutdown time. An accepted FPS request or metadata value is not a measurement of delivered frames.

Capture uses OpenCV. The Mac built-in camera used for this project supports up to 30 FPS; requesting a higher rate cannot make it deliver additional frames. Window rendering and processing can reduce preview FPS below the capture rate.

### Saved videos

```bash
.venv/bin/python live_stylizer.py \
  --input input.mp4 --output tmp/stylized.mp4 \
  --face-mode oval --cloth-color '#20A4F3' --pattern plain
```

Files are processed sequentially with synchronous inference on every frame. The output uses the `mp4v` codec and source FPS metadata. Audio is not copied. Live recording uses a fixed nominal FPS; dropped camera frames can change playback duration.

### Privacy limits

This is a demonstration, not a guarantee that every face remains hidden. A newly appearing face can be visible before detection. Complete occlusion, fast movement, unusual angles, or poor lighting can cause missed detections or tracking drift. An expired track no longer has a cover. Blur and mosaic can retain identifying information; the opaque modes remove detail only within the covered region.

The project uses pretrained models and has no training pipeline. Downloading a dataset alone does not improve those models; further improvements need evaluation and, where appropriate, retraining on representative occlusion and motion examples.

## Password-based video encryption

```bash
.venv/bin/python main.py encrypt input.mp4 tmp/output.venc
.venv/bin/python main.py info tmp/output.venc
.venv/bin/python main.py decrypt tmp/output.venc tmp/recovered.mkv
```

Omit `--password` to enter it through a hidden prompt. Encryption asks for confirmation. For automated demos, both encryption and decryption accept `--password 'demo-password'`.

`main.py` derives a 256-bit key with Scrypt and encrypts each PNG-encoded frame independently with AES-256-GCM. Each frame has its own nonce; the container header and frame index are included as authenticated associated data. Wrong passwords or modified authenticated records cause decryption to fail. Decryption writes lossless FFV1 video, requiring an OpenCV build with FFmpeg/FFV1 support.

The prototype stores decoded video frames, not the original compressed file, audio, or all source metadata. The header is readable without a password. VENC v1 authenticates frame records but does not authenticate the final frame count/end marker, so it does not guarantee detection of every shortened container. A failed operation can leave a partial output file. Use a new output path for each run.

## Experimental chaos lab

```bash
.venv/bin/python chaos_lab.py input.mp4 tmp/chaos-demo/
```

The experiment reads the first video frame and creates:

- `plain.png`, `scrambled.png`, and `recovered.png`.
- `comparison.png` showing the three images side by side.
- `arnold_cat_map.png` from a square crop.
- `metrics.json` with entropy, MSE, PSNR, NPCR, UACI, correlation, and recovery metrics.

Options include `--seed` (default `0.3141592653`) and `--iterations` (default `8`). The scrambler combines logistic-map permutations and an XOR mask. It is an educational experiment; its image metrics do not establish cryptographic security.

## Project structure

| File | Responsibility |
| --- | --- |
| `live_stylizer.py` | CLI, startup validation, preview loop, recording, and performance reporting. |
| `live_capture.py` | Mac camera/file input, camera settings, and latest-frame reader. |
| `vision_rendering.py` | Face effects, garment tint/patterns, and status panels. |
| `vision_tracking.py` | Face association, optical flow, clothing smoothing, and model engine. |
| `realtime_vision.py` | Asynchronous inference and alignment of delayed observations. |
| `vision_models.py` | Model downloads and checksum verification. |
| `main.py` | VENC container encryption, decryption, and metadata CLI. |
| `chaos_lab.py` | Chaos scrambling and measurement demo. |
| `tests/` | Offline unit tests and local-model integration checks. |
| `models/` | Downloaded vision models and the YuNet license. |
| `tmp/` | Ignored diagnostic artifacts. |

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python live_stylizer.py --help
```

The suite checks occlusion retention, partial detections, oversized observations, multiple faces, opaque rendering at image edges, clothing vetoes, Mac camera/file input selection, camera backlog/shutdown, asynchronous inference, stale results, and startup validation. The real-model integration test needs the local models and MediaPipe; it skips when they are absent. Tests do not open a camera.

## Troubleshooting

- **Camera cannot open or delivers no frames:** verify Camera permission for the running app, close other camera apps, and use `--input 0` for live input.
- **An old cover remains:** press **R** to reset immediately, or allow the track to expire.
- **Model download fails:** rerun `vision_models.py` with internet access. A partial file is never treated as a verified model.
- **Imports fail in PyCharm:** select this project's `.venv/bin/python`, then run `uv sync --locked` from the project folder.
- **Preview is slower than capture:** inspect `--profile`, reduce `--preview-width` for window overhead, or reduce `--max-width` for processing cost. Reducing the latter also reduces output resolution.

## Model sources

Face detection uses [`face_detection_yunet_2023mar.onnx` from OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). Its MIT license is retained in [`models/YUNET_LICENSE`](models/YUNET_LICENSE).

Garment segmentation uses Google's [`selfie_multiclass_256x256.tflite`](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) with background, hair, body skin, face skin, clothes, and accessories classes. Exact download URLs and checksums are defined in `vision_models.py`.
