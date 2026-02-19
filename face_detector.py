from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

# CRITICAL: Set up CUDA paths BEFORE importing TensorFlow
# This is needed because TensorFlow doesn't always find CUDA libraries automatically
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
cuda_bin = os.path.join(cuda_path, "bin")
cudnn_path = os.path.join(os.path.expanduser('~'), '.local', 'lib', 'python3.12', 'site-packages', 'nvidia_cudnn_cu12')

# Add CUDA bins to PATH
if os.path.exists(cuda_bin):
    os.environ['PATH'] = cuda_bin + ';' + os.environ.get('PATH', '')
    os.environ['CUDA_PATH'] = cuda_path
    os.environ['CUDA_HOME'] = cuda_path
    os.environ['CUDA_TOOLKIT_ROOT_DIR'] = cuda_path
    print(f"[INFO] CUDA paths configured: {cuda_path}")

from deepface import DeepFace

# Configure DeepFace home directory for PyInstaller
# CRITICAL: Always use user home directory for models, NOT bundle
# This ensures we use the downloaded models from ~/.deepface
deepface_home = os.path.join(os.path.expanduser('~'), '.deepface')

if getattr(sys, 'frozen', False):
    # Running as compiled executable
    print(f"[INFO] PyInstaller executable detected")
    print(f"[INFO] Using DeepFace models from user home: {deepface_home}")
else:
    # Running from source
    print(f"[INFO] Running from source")
    print(f"[INFO] Using DeepFace models from: {deepface_home}")

# Set DEEPFACE_HOME environment variable
os.environ['DEEPFACE_HOME'] = deepface_home

# Enable GPU acceleration if available
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPU acceleration enabled: {len(gpus)} GPU(s) detected")
        except RuntimeError as e:
            print(f"[WARNING] Could not configure GPU: {e}")
    else:
        print(f"[INFO] No GPUs detected, using CPU")
except Exception as e:
    print(f"[WARNING] GPU detection failed: {e}")
if not os.path.exists(deepface_home):
    print(f"[WARNING] DeepFace home directory does not exist: {deepface_home}")
    print(f"[WARNING] Models will be downloaded automatically on first detection")
else:
    print(f"[INFO] DeepFace home directory found")
    weights_dir = os.path.join(deepface_home, 'weights')
    if os.path.exists(weights_dir):
        print(f"[INFO] Weights directory exists with models")
    else:
        print(f"[WARNING] Weights directory not found, may need model download")


@dataclass
class DetectionBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    mask: Optional[np.ndarray] = None  # Pre-computed mask for performance


class RobustFaceCensor:
    def __init__(
        self,
        min_confidence: float = 0.01,  # ULTRA-SENSITIVE: Detects partial faces, individual features
        mode: str = "pixel",
        detect_every_n_frames: int = 6,  # Detect every 6 frames by default
        max_detection_size: int = 1280,
        pixelation_strength: float = 15.0,  # 0-30 pixels (0=sharp, 30=very pixelated)
    ) -> None:
        print(f"[INFO] Initializing RobustFaceCensor")
        print(f"[INFO] Mode: {mode}, Min confidence: {min_confidence} (ULTRA-SENSITIVE)")
        print(f"[INFO] Detect every {detect_every_n_frames} frames")
        print(f"[INFO] Pixelation strength: {pixelation_strength} pixels (0-30)")
        print(f"[INFO] Expansion factor: 50% (to cover hair, edges, partial faces)")
        
        self.mode = mode
        self.min_confidence = min_confidence
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self.max_detection_size = max(480, int(max_detection_size))
        self.pixelation_strength = max(0.0, min(30.0, pixelation_strength))  # Enforce 0-30 range

        # Motion tracking / position prediction
        self._frame_counter = 0
        self._last_detections: List[DetectionBox] = []  # Keep last detected positions
        self._frame_since_detection = 0  # Counter for reusing last detection
        
        print(f"[INFO] RobustFaceCensor initialized successfully")

    def close(self) -> None:
        pass

    def _compute_mask(self, w: int, h: int) -> np.ndarray:
        """Pre-compute oval mask for faster processing during censor."""
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x = w // 2
        center_y = h // 2
        axis_x = int(w * 0.45)
        axis_y = int(h * 0.50)
        
        cv2.ellipse(
            mask,
            center=(center_x, center_y),
            axes=(axis_x, axis_y),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1
        )
        
        return mask

    def _detect_once(self, frame_bgr: np.ndarray) -> List[DetectionBox]:
        height, width = frame_bgr.shape[:2]
        boxes: List[DetectionBox] = []

        try:
            deepface_home = os.environ.get('DEEPFACE_HOME', 'NOT SET')
            if deepface_home == 'NOT SET':
                raise RuntimeError("DEEPFACE_HOME environment variable is not set")
            
            detections = DeepFace.extract_faces(
                img_path=frame_bgr,
                detector_backend="retinaface",
                enforce_detection=False,
                align=False,
            )
            
            if len(detections) > 0:
                print(f"[INFO] Detected {len(detections)} face(s)")

            for detection in detections:
                facial_area = detection.get("facial_area", {})
                x = int(facial_area.get("x", 0))
                y = int(facial_area.get("y", 0))
                w = int(facial_area.get("w", 0))
                h = int(facial_area.get("h", 0))
                confidence = float(detection.get("confidence", 1.0))
                if confidence <= 0:
                    confidence = 0.95  # Default high confidence if not available

                # ULTRA-SENSITIVE: Expand bbox by 50% to cover hair, edges, partial faces
                # This ensures we catch even partial detections (just a mouth, nose, jaw, etc)
                expand_factor = 0.50
                expand_w = int(w * expand_factor)
                expand_h = int(h * expand_factor)
                
                x = max(0, x - expand_w // 2)
                y = max(0, y - expand_h // 2)
                w = w + expand_w
                h = h + expand_h

                # Bounds check after expansion
                if x >= width or y >= height or w <= 0 or h <= 0:
                    continue
                if x + w > width:
                    w = width - x
                if y + h > height:
                    h = height - y

                # ULTRA-SENSITIVE MODE: Accept ALL detections
                # Even with very low confidence (0.01), we censor.
                # This ensures we catch partial faces, individual features (mouth, nose, etc.)
                if confidence >= self.min_confidence:
                    mask = self._compute_mask(w, h)
                    boxes.append(DetectionBox(x, y, max(1, w), max(1, h), confidence, mask))

        except Exception as e:
            import traceback
            print(f"[ERROR] Face detection failed: {str(e)}")
            traceback.print_exc()

        return boxes

    def _apply_censor(self, frame: np.ndarray, box: DetectionBox) -> None:
        """Apply censorship using pre-computed mask."""
        x, y, w, h = box.x, box.y, box.w, box.h
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return

        # Create pixelated version
        # Slider 0 = blocos de 2px (bem nítido)
        # Slider 30 = blocos de 50px (muito pixelado)
        block_size = max(2, int(2 + self.pixelation_strength * 1.6))
        pixel_w = max(1, w // block_size)
        pixel_h = max(1, h // block_size)
        downscaled = cv2.resize(roi, (pixel_w, pixel_h), interpolation=cv2.INTER_LINEAR)
        censored = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)

        # Use pre-computed mask
        if box.mask is not None:
            mask_img = box.mask
        else:
            mask_img = np.zeros((h, w), dtype=np.uint8)
            center_x = w // 2
            center_y = h // 2
            axis_x = int(w * 0.45)
            axis_y = int(h * 0.50)
            cv2.ellipse(mask_img, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
        
        # Apply mask with blending
        mask_3d = np.stack([mask_img] * 3, axis=2).astype(np.float32) / 255.0
        blended = (censored.astype(np.float32) * mask_3d) + (roi.astype(np.float32) * (1 - mask_3d))
        blended = np.uint8(np.clip(blended, 0, 255))
        
        frame[y : y + h, x : x + w] = blended

    def censor_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """Detect and censor faces, reusing last detection every N frames."""
        frame = frame_bgr.copy()
        self._frame_counter += 1
        self._frame_since_detection += 1

        # Detect new faces every N frames
        if self._frame_counter == 1 or self._frame_since_detection >= self.detect_every_n_frames:
            frame_h, frame_w = frame.shape[:2]
            max_dim = max(frame_w, frame_h)
            
            # Downscale for faster detection if needed
            detect_frame = frame
            if max_dim > self.max_detection_size:
                scale = self.max_detection_size / max_dim
                detect_w = max(1, int(frame_w * scale))
                detect_h = max(1, int(frame_h * scale))
                detect_frame = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_AREA)

            # Run detection
            detections = self._detect_once(detect_frame)

            # Scale back up if downscaled
            if max_dim > self.max_detection_size:
                scale = self.max_detection_size / max_dim
                inv_scale = 1.0 / scale
                scaled_detections: List[DetectionBox] = []
                for box in detections:
                    new_w = max(1, int(box.w * inv_scale))
                    new_h = max(1, int(box.h * inv_scale))
                    
                    # Scale the mask if available
                    scaled_mask = None
                    if box.mask is not None:
                        scaled_mask = cv2.resize(box.mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    scaled_detections.append(
                        DetectionBox(
                            x=int(box.x * inv_scale),
                            y=int(box.y * inv_scale),
                            w=new_w,
                            h=new_h,
                            score=box.score,
                            mask=scaled_mask,
                        )
                    )
                detections = scaled_detections

            # Store detections for reuse
            self._last_detections = detections
            self._frame_since_detection = 0
        
        # Use stored detections (either new or reused from previous frame)
        for box in self._last_detections:
            self._apply_censor(frame, box)

        return frame, len(self._last_detections)