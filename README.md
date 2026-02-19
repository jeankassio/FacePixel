# FacePixel

FacePixel is a desktop application for **automatic face censoring in MP4 videos** with a modern GUI.

It uses:
- **InsightFace RetinaFace** as the primary detector (GPU-accelerated when CUDA is available).
- **DeepFace RetinaFace** as a fallback detector.
- Pixelation-based censoring with real-time preview.

## Features

- Load and preview MP4 videos in-app.
- Adjustable pixelation strength (`0` to `30`).
- Adjustable detection frequency (`detect every N frames`) to balance speed vs. accuracy.
- Live processing preview and progress updates.
- Cancel processing and automatically delete partial output.
- Output naming with custom suffix (example: `video_censored.mp4`).

## How It Works

1. You select an input MP4.
2. The app builds an output path using the selected suffix.
3. Frames are processed through `RobustFaceCensor`.
4. Faces are detected periodically (based on `detect_every_n_frames`) and reused between detection cycles for performance.
5. Pixelation is applied to detected face regions.
6. A new MP4 is written and loaded in the preview player after completion.

## Tech Stack

- Python 3
- OpenCV (`opencv-python`)
- PyAV (`av`)
- NumPy
- Pillow
- ttkbootstrap + Tkinter
- PyTorch
- InsightFace + ONNX Runtime
- DeepFace + TensorFlow (`tf-keras`)
- PyInstaller (optional, for executable build)

## Project Structure

- `main.py` - GUI, media playback panels, app lifecycle, processing orchestration.
- `video_processor.py` - frame-by-frame processing pipeline and output writer.
- `face_detector_pytorch.py` - primary detector/censor implementation (InsightFace).
- `face_detector.py` - fallback detector/censor implementation (DeepFace).
- `download_models.py` - pre-downloads InsightFace and DeepFace model files.
- `FacePixel.spec` - PyInstaller configuration for packaging.
- `requirements.txt` - Python dependencies.
- `install.bat` / `install.sh` - quick install scripts.
- `run_app.bat` - create venv, install deps, and run app on Windows.
- `build_exe.bat` - build Windows executable.

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA (optional, for faster detection).

FacePixel still works on CPU, but processing can be significantly slower.

## Installation

### Option A: Manual setup

1. Create and activate a virtual environment.
2. Install PyTorch (CUDA build if needed):
   - CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   - CPU: `pip install torch torchvision`
3. Install remaining dependencies:
   - `pip install -r requirements.txt`
4. run `python3 main.py`

### Option B: Windows quick run

- Run: `run_app.bat`

This script creates `.venv`, installs dependencies, and launches the app.

## Model Storage

Models are stored in user directories (not bundled into the executable):
- InsightFace: `~/.insightface`
- DeepFace: `~/.deepface`

This keeps build size smaller and allows model reuse/updates.

## Building an Executable (Windows)

```bat
build_exe.bat
```

Or manually:

```bash
pyinstaller FacePixel.spec
```

The generated executable is expected in:
- `dist/FacePixel.exe`

## UI Controls

- **Pixelation slider**: controls block size intensity.
- **Detect every N frames**: lower values improve tracking/accuracy, higher values improve speed.
- **Start Censoring**: starts background processing.
- **Cancel**: stops processing and removes partial output.
- **Source/Censored players**:
  - Play/Pause
  - Rewind 5s
  - Forward 5s
  - Seek bar

## Notes and Limitations

- Input selection is restricted to MP4 in the GUI.
- Processing speed depends on resolution, FPS, hardware, and selected detection interval.
- If no GPU is available, the app falls back to CPU execution.
- The detector uses aggressive sensitivity settings to catch partial/occluded faces.

## Troubleshooting

- If models fail to download on first run, check internet connectivity and run:
  - `python download_models.py`
- If GPU is not detected, verify CUDA-compatible PyTorch and drivers.
- If build fails, ensure all dependencies from `requirements.txt` are installed in the active environment.

## License

MIT License. See `LICENSE`.
