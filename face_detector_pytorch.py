from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

@dataclass
class DetectedFace:
    x: int
    y: int
    w: int
    h: int
    confidence: float


class RobustFaceCensor:
    """Face detector and censor using InsightFace RetinaFace (GPU accelerated)."""
    
    def __init__(
        self,
        min_confidence: float = 0.01,
        mode: str = "pixel",
        detect_every_n_frames: int = 6,
        max_detection_size: int = 1280,
        pixelation_strength: float = 15.0,
        device: str = None,
    ) -> None:
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"[INFO] Initializing RobustFaceCensor with InsightFace RetinaFace")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Mode: {mode}, Min confidence: {min_confidence} (ULTRA-SENSITIVE)")
        print(f"[INFO] Detect every {detect_every_n_frames} frames")
        print(f"[INFO] Pixelation strength: {pixelation_strength} pixels (0-30)")
        print(f"[INFO] Expansion: -20% (shrinks to censor ONLY inside face boundaries)")
        
        self.mode = mode
        self.min_confidence = min_confidence
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self.max_detection_size = max(480, int(max_detection_size))
        self.pixelation_strength = max(0.0, min(30.0, pixelation_strength))
        
        # Initialize InsightFace with RetinaFace detector
        try:
            print("[INFO] Loading InsightFace RetinaFace model...")
            # Use the appropriate provider for GPU/CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            
            self.app = FaceAnalysis(
                name='buffalo_l',  # or 'buffalo_sc' for smaller/faster
                providers=providers,
                allowed_modules=['detection']  # Only load detection, not recognition
            )
            # Ultra-low thresholds for maximum detection of partially covered/difficult faces
            # Higher resolution (800x800) for better detail capture
            self.app.prepare(
                ctx_id=0 if self.device == 'cuda' else -1, 
                det_size=(800, 800),  # Higher resolution for better detection
                det_thresh=0.01  # Maximum sensitivity (default: 0.5)
            )
            print(f"[INFO] InsightFace RetinaFace loaded on {self.device}")
            print(f"[INFO] Detection threshold: 0.1 (maximum sensitivity)")
            print(f"[INFO] Detection size: 800x800 (high resolution)")
        except Exception as e:
            print(f"[ERROR] Failed to load InsightFace model: {e}")
            raise
        
        # Tracking for frame reuse
        self._frame_counter = 0
        self._frame_since_detection = 0
        self._last_faces: List[DetectedFace] = []
        self._has_valid_cache = False
        
        # Performance counters
        self.detection_count = 0
        self.reuse_count = 0
    
    def _downscale_for_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Downscale frame if needed for faster detection."""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.max_detection_size:
            return frame, 1.0
        
        scale = self.max_detection_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        downscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return downscaled, scale
    
    def _detect_faces_insightface(self, frame: np.ndarray) -> List[DetectedFace]:
        """Detect faces using InsightFace RetinaFace (GPU accelerated)."""
        try:
            # InsightFace expects BGR format (OpenCV default)
            faces_data = self.app.get(frame)
            
            if not faces_data or len(faces_data) == 0:
                return []
            
            faces = []
            for face in faces_data:
                # Get bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Get detection score
                confidence = float(face.det_score)
                
                # Ultra-low confidence filter for maximum detection
                # Allows partially covered faces (hair, hands, angles, etc.)
                if confidence < 0.1:  # Maximum permissive threshold
                    continue
                
                w = x2 - x1
                h = y2 - y1
                
                # Only reject extremely small detections
                if w < 10 or h < 10:
                    continue
                
                faces.append(DetectedFace(
                    x=x1,
                    y=y1,
                    w=w,
                    h=h,
                    confidence=confidence
                ))
            
            return faces
        
        except Exception as e:
            print(f"[WARNING] InsightFace detection failed: {e}")
            return []
    
    def detect_faces(self, frame: np.ndarray) -> List[DetectedFace]:
        """Detect faces in a frame with caching and reuse logic."""
        self._frame_counter += 1
        self._frame_since_detection += 1
        
        # Check if we should perform a new detection
        should_detect = (
            self._frame_counter == 1 or 
            self._frame_since_detection >= self.detect_every_n_frames
        )
        
        if should_detect:
            # Perform detection (InsightFace works with BGR directly)
            downscaled, scale = self._downscale_for_detection(frame)
            
            faces = self._detect_faces_insightface(downscaled)
            
            # Scale back coordinates
            if scale != 1.0:
                for face in faces:
                    face.x = int(face.x / scale)
                    face.y = int(face.y / scale)
                    face.w = int(face.w / scale)
                    face.h = int(face.h / scale)
            
            # Apply NEGATIVE expansion (-20%) to shrink and censor ONLY inside face
            # This prevents censoring outside the face boundaries
            expand_factor = -0.02  # Negative = shrink inward
            h, w = frame.shape[:2]
            
            expanded_faces = []
            for face in faces:
                # Negative expansion shrinks the box
                expand_w = int(face.w * abs(expand_factor))
                expand_h = int(face.h * abs(expand_factor))
                
                # Shrink inward: move start position in, reduce size
                x = max(0, face.x + expand_w)
                y = max(0, face.y + expand_h)
                face_w = max(10, face.w - 2 * expand_w)
                face_h = max(10, face.h - 2 * expand_h)
                
                # Ensure within frame bounds
                face_w = min(w - x, face_w)
                face_h = min(h - y, face_h)
                
                expanded_faces.append(DetectedFace(
                    x=x,
                    y=y,
                    w=face_w,
                    h=face_h,
                    confidence=face.confidence
                ))
            
            self._last_faces = expanded_faces
            self._has_valid_cache = True
            self._frame_since_detection = 0
            self.detection_count += 1
            
            return expanded_faces
        else:
            # Reuse cached faces
            self.reuse_count += 1
            return self._last_faces if self._has_valid_cache else []
    
    def _apply_pixelation(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Apply pixelation to a region in the frame."""
        # Calculate block size based on pixelation strength
        # 0 pixels = 2px blocks (sharp), 30 pixels = 50px blocks (very pixelated)
        block_size = max(2, int(2 + self.pixelation_strength * 1.6))
        
        # Extract the region
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return
        
        # Calculate downscaled dimensions
        pixel_w = max(1, w // block_size)
        pixel_h = max(1, h // block_size)
        
        # Downscale and upscale to create pixelation effect
        try:
            downscaled = cv2.resize(roi, (pixel_w, pixel_h), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = pixelated
        except Exception as e:
            print(f"[WARNING] Pixelation failed for region ({x},{y},{w},{h}): {e}")
    
    def _apply_blur(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Apply Gaussian blur to a region."""
        kernel_size = max(21, (w // 5) | 1)
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return
        
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        frame[y:y+h, x:x+w] = blurred
    
    def censor(self, frame: np.ndarray, faces: Optional[List[DetectedFace]] = None) -> np.ndarray:
        """Apply censoring to detected faces in a frame."""
        if faces is None:
            faces = self.detect_faces(frame)
        
        if not faces:
            return frame
        
        result = frame.copy()
        
        for face in faces:
            if self.mode == "pixel":
                self._apply_pixelation(result, face.x, face.y, face.w, face.h)
            elif self.mode == "blur":
                self._apply_blur(result, face.x, face.y, face.w, face.h)
        
        return result
    
    def censor_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """Detect and censor faces in a frame, returning (censored_frame, face_count)."""
        faces = self.detect_faces(frame_bgr)
        censored = self.censor(frame_bgr, faces)
        return censored, len(faces)
    
    def get_stats(self) -> str:
        """Get performance statistics."""
        total = self.detection_count + self.reuse_count
        if total == 0:
            return "No frames processed yet"
        
        reuse_pct = (self.reuse_count / total) * 100
        return (
            f"Processed {total} frames: "
            f"{self.detection_count} detections, "
            f"{self.reuse_count} reuses ({reuse_pct:.1f}%)"
        )
    
    def close(self) -> None:
        """Clean up resources (PyTorch handles cleanup automatically)."""
        pass
