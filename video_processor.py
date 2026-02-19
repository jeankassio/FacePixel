from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

# Try PyTorch detector first (GPU accelerated), fall back to DeepFace if not available
try:
    from face_detector_pytorch import RobustFaceCensor
    print("[INFO] Using InsightFace RetinaFace detector (GPU accelerated)")
    USE_PYTORCH = True
except ImportError:
    from face_detector import RobustFaceCensor
    print("[INFO] Using DeepFace detector (CPU only)")
    USE_PYTORCH = False


PreviewCallback = Callable[[np.ndarray], None]
ProgressCallback = Callable[[int, int], None]
CancelCheckCallback = Callable[[], bool]


def _choose_speed_profile(width: int, height: int, fps: float) -> tuple[int, int, int, int]:
    """Choose speed profile based on video resolution and frame rate.
    
    Since DeepFace/TensorFlow on CPU is very slow (~2-3 seconds per detection),
    we need AGGRESSIVE frame skipping to make processing feasible.
    """
    pixels = width * height
    
    # Be very aggressive with frame skipping for CPU
    # This trades quality for speed - detects only every N frames
    max_detection_size = 1280
    detect_every_n_frames = 1

    # Adjust based on resolution - higher res = more skipping needed
    if pixels >= 3840 * 2160:  # 4K - skip a LOT
        max_detection_size = 512
        detect_every_n_frames = 16  # Detect only 2 times per second
    elif pixels >= 2560 * 1440:  # 2K
        max_detection_size = 640
        detect_every_n_frames = 12
    elif pixels >= 1920 * 1080:  # Full HD
        max_detection_size = 768
        detect_every_n_frames = 10  # Detect 3 times per second at 30fps
    elif pixels >= 1280 * 720:  # HD
        max_detection_size = 896
        detect_every_n_frames = 8  # Detect 3-4 times per second
    else:  # Lower res
        max_detection_size = 1024
        detect_every_n_frames = 6

    # Update based on frame rate if very high
    if fps > 60:
        detect_every_n_frames = max(detect_every_n_frames, 12)
    elif fps > 30:
        pass  # Keep as is

    # Preview and progress updates
    preview_every_n_frames = max(1, int(round(fps / 8.0)))
    progress_every_n_frames = max(1, int(round(fps / 3.0)))

    
    return max_detection_size, detect_every_n_frames, preview_every_n_frames, progress_every_n_frames


def build_output_path(input_path: str, suffix: str) -> str:
    src = Path(input_path)
    clean_suffix = suffix.strip() or "censored"
    return str(src.with_name(f"{src.stem}_{clean_suffix}{src.suffix}"))


def censor_video(
    input_path: str,
    output_path: str,
    on_preview: Optional[PreviewCallback] = None,
    on_progress: Optional[ProgressCallback] = None,
    pixelation_strength: float = 5.0,
    detect_every_n_frames: Optional[int] = None,
    cancel_check: Optional[CancelCheckCallback] = None,
) -> str:
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open the selected video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[DEBUG] Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the output MP4 file.")

    cv2.setUseOptimized(True)

    max_detection_size, auto_detect_every_n_frames, preview_every_n_frames, progress_every_n_frames = _choose_speed_profile(
        width,
        height,
        fps,
    )
    
    # Use user-specified value if provided, otherwise use automatic profile
    if detect_every_n_frames is None:
        detect_every_n_frames = auto_detect_every_n_frames
    
    print(f"[DEBUG] Speed profile: max_detection={max_detection_size}, detect_every={detect_every_n_frames}, preview_every={preview_every_n_frames}")

    detector = RobustFaceCensor(
        mode="pixel",
        detect_every_n_frames=detect_every_n_frames,
        max_detection_size=max_detection_size,
        pixelation_strength=pixelation_strength,
    )
    frame_index = 0
    detection_log_interval = max(1, total_frames // 20)  # Log detection stats every 5%
    write_errors = 0

    try:
        while True:
            # Check for cancellation
            if cancel_check is not None and cancel_check():
                print("[DEBUG] Processing cancelled by user")
                break
            
            ok, frame = capture.read()
            if not ok:
                break

            censored_frame, face_count = detector.censor_frame(frame)
            
            # Write frame and check for errors
            write_ok = writer.write(censored_frame)
            if not write_ok:
                write_errors += 1
                if write_errors <= 3:
                    print(f"[WARNING] Frame {frame_index} write failed")
            
            frame_index += 1

            # Log detection stats periodically (every 5% of video)
            if frame_index % detection_log_interval == 0 or frame_index <= 3:
                print(f"[DEBUG] Frame {frame_index}/{total_frames}: {face_count} faces detected")

            if on_preview is not None and (frame_index % preview_every_n_frames == 0):
                on_preview(censored_frame)

            should_emit_progress = (
                frame_index == 1
                or frame_index == total_frames
                or (frame_index % progress_every_n_frames == 0)
            )
            if on_progress is not None and should_emit_progress:
                on_progress(frame_index, total_frames)

    finally:
        detector.close()
        writer.release()
        capture.release()

    return output_path