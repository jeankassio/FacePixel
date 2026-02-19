from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from tkinter import BOTH, HORIZONTAL, LEFT, RIGHT, X, Button, Canvas, Entry, Frame, Label, Scale, StringVar, filedialog, messagebox

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import av
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch

from video_processor import build_output_path, censor_video


def ensure_deepface_models() -> bool:
    """Ensure DeepFace models are available.
    
    Returns:
        True if models are available or were successfully downloaded.
        False if models could not be obtained.
    """
    deepface_home = os.path.join(os.path.expanduser('~'), '.deepface')
    weights_dir = os.path.join(deepface_home, 'weights')
    retinaface_model = os.path.join(weights_dir, 'retinaface.h5')
    
    # Check if models already exist
    if os.path.exists(retinaface_model):
        print(f"[OK] DeepFace models found at {retinaface_model}")
        return True
    
    # Try to download models if they don't exist
    print(f"[INFO] DeepFace models not found. Attempting to download...")
    print(f"[INFO] This may take a minute on first run...")
    
    try:
        from deepface import DeepFace
        
        # Create a dummy image and trigger model download
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # This will automatically download models to ~/.deepface
        print(f"[INFO] Downloading RetinaFace model (this happens automatically)...")
        DeepFace.extract_faces(
            img_path=test_img,
            detector_backend="retinaface",
            enforce_detection=False,
            align=False
        )
        
        # Verify download
        if os.path.exists(retinaface_model):
            print(f"[OK] Models successfully downloaded to {deepface_home}")
            return True
        else:
            print(f"[WARNING] Model download completed but file not found at expected location")
            # Models might still be usable, so return True
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to download models: {e}")
        return False


class SourceVideoPlayerPanel:
    def __init__(self, root: ttk.Window, parent: Frame, title: str) -> None:
        self.root = root
        self.container = Frame(parent, relief="groove", borderwidth=1, height=350)
        self.container.pack(side=LEFT, fill=BOTH, expand=True, padx=6, pady=6)
        self.container.pack_propagate(False)  # Impede que o conteúdo redimensione o container

        self.title_label = ttk.Label(self.container, text=title, anchor="w", font=("Segoe UI", 10, "bold"))
        self.title_label.pack(fill=X, padx=8, pady=(8, 4))

        self.video_label = Label(self.container, text="No video loaded", relief="sunken")
        self.video_label.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))

        controls = Frame(self.container)
        controls.pack(fill=X, padx=8, pady=(0, 8))

        self.play_button = ttk.Button(controls, text="▶ Play", width=12, command=self.toggle_play, bootstyle="success")
        self.play_button.pack(side=LEFT)

        self.rewind_button = ttk.Button(controls, text="⏪ -5s", command=self.rewind_5s, bootstyle="secondary")
        self.rewind_button.pack(side=LEFT, padx=(8, 0))

        self.forward_button = ttk.Button(controls, text="⏩ +5s", command=self.forward_5s, bootstyle="secondary")
        self.forward_button.pack(side=LEFT, padx=(8, 0))

        self.time_label = ttk.Label(controls, text="00:00 / 00:00")
        self.time_label.pack(side=RIGHT)

        self.seek_scale = ttk.Scale(
            self.container,
            from_=0,
            to=0,
            orient=HORIZONTAL,
            command=self._on_seek,
        )
        self.seek_scale.pack(fill=X, padx=8, pady=(0, 8))

        self.progress_label = ttk.Label(self.container, text="Progress: 0/0 (0.0%)", anchor="w")
        self.progress_label.pack(fill=X, padx=8, pady=(0, 8))

        self.video_path: str | None = None
        self.player = None
        self.stream = None
        self.decode_iter = None
        self.fps = 30.0
        self.duration_sec = 0.0
        self.total_frames = 0
        self.current_sec = 0.0
        self.current_frame = 0
        self.is_playing = False
        self.user_can_control = True
        self.ignore_seek_event = False
        self.current_image = None
        self.target_ui_fps = 30.0
        self._player_lock = threading.Lock()
        self._stop_worker = threading.Event()
        self._frame_queue: Queue = Queue(maxsize=3)
        self._worker_thread: threading.Thread | None = None
        self._last_emitted_sec = 0.0

        self.root.after(30, self._tick)

    def is_loaded(self) -> bool:
        return self.player is not None and self.stream is not None

    def set_title(self, text: str) -> None:
        self.title_label.configure(text=text)

    def set_controls_enabled(self, enabled: bool) -> None:
        self.user_can_control = enabled
        state = "normal" if enabled else "disabled"
        self.play_button.configure(state=state)
        self.rewind_button.configure(state=state)
        self.forward_button.configure(state=state)
        self.seek_scale.configure(state=state)

    def release(self) -> None:
        self._stop_worker.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.8)
        self._worker_thread = None
        self.is_playing = False

        with self._player_lock:
            if self.player is not None:
                self.player.close()
            self.player = None
            self.stream = None
            self.decode_iter = None

    def stop(self) -> None:
        self.is_playing = False
        self.play_button.configure(text="Play")

    def clear_display(self, text: str) -> None:
        self.video_label.configure(text=text, image="")
        self.current_image = None
        self.current_sec = 0.0
        self.current_frame = 0
        self.total_frames = 0
        self.duration_sec = 0.0
        self._update_time_ui()

    def load_video(self, path: str) -> bool:
        self.release()

        try:
            player = av.open(path)
        except Exception:
            self.clear_display("Failed to load video")
            return False

        video_streams = [s for s in player.streams if s.type == "video"]
        if not video_streams:
            player.close()
            self.clear_display("No video stream found")
            return False

        stream = video_streams[0]
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        self.fps = rate if rate > 0 else 30.0

        duration_sec = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration_sec = float(stream.duration * stream.time_base)
        elif player.duration is not None:
            duration_sec = float(player.duration) / 1_000_000.0

        self.duration_sec = max(0.0, duration_sec)
        self.total_frames = int(self.duration_sec * self.fps) if self.duration_sec > 0 else 0
        self.current_sec = 0.0
        self.current_frame = 0
        self.video_path = path
        self._last_emitted_sec = 0.0

        with self._player_lock:
            self.player = player
            self.stream = stream
            self.decode_iter = self.player.decode(video=0)

        self.ignore_seek_event = True
        self.seek_scale.configure(to=max(0, int(self.duration_sec * 1000)))
        self.seek_scale.set(0)
        self.ignore_seek_event = False

        self._seek_to_seconds(0.0, render_now=True)
        self.stop()
        return True

    def toggle_play(self) -> None:
        if not self.user_can_control or not self.is_loaded():
            return
        self.is_playing = not self.is_playing
        self.play_button.configure(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self._ensure_worker()

    def forward_5s(self) -> None:
        if not self.user_can_control or not self.is_loaded():
            return
        self._seek_to_seconds(self.current_sec + 5.0, render_now=True)

    def rewind_5s(self) -> None:
        if not self.user_can_control or not self.is_loaded():
            return
        self._seek_to_seconds(self.current_sec - 5.0, render_now=True)

    def _on_seek(self, value: str) -> None:
        if self.ignore_seek_event or not self.user_can_control:
            return
        if self.is_playing:
            return
        if not self.is_loaded():
            return
        target = float(value) / 1000.0
        if abs(target - self.current_sec) < 0.2:
            return
        self._seek_to_seconds(target, render_now=True)

    def _tick(self) -> None:
        try:
            while True:
                frame_time, rgb_frame = self._frame_queue.get_nowait()
                self.current_sec = frame_time
                self.current_frame = int(self.current_sec * self.fps)
                self._render_rgb_frame(rgb_frame)
                self._update_time_ui()
        except Empty:
            pass
        self.root.after(max(15, int(1000 / self.target_ui_fps)), self._tick)

    def _ensure_worker(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_worker.clear()
        self._worker_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker_thread.start()

    def _playback_worker(self) -> None:
        frame_interval = 1.0 / max(1.0, self.target_ui_fps)
        while not self._stop_worker.is_set():
            if not self.is_playing:
                time.sleep(0.01)
                continue

            with self._player_lock:
                if self.player is None or self.decode_iter is None:
                    time.sleep(0.01)
                    continue
                frame = self._decode_next_frame_locked()

            if frame is None:
                self.is_playing = False
                self.root.after(0, self.stop)
                continue

            frame_time = frame.time if frame.time is not None else self._last_emitted_sec + (1.0 / self.fps)
            frame_time = float(frame_time)
            if frame_time <= self._last_emitted_sec:
                frame_time = self._last_emitted_sec + (1.0 / max(1.0, self.fps))
            self._last_emitted_sec = frame_time

            rgb = frame.to_ndarray(format="rgb24")
            rgb = self._prepare_preview_rgb(rgb)

            try:
                self._frame_queue.put_nowait((float(frame_time), rgb))
            except Exception:
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._frame_queue.put_nowait((float(frame_time), rgb))
                except Exception:
                    pass

            time.sleep(frame_interval)

    def _decode_next_frame_locked(self):
        if self.decode_iter is None:
            return None
        try:
            return next(self.decode_iter)
        except StopIteration:
            return None
        except Exception:
            return None

    def _seek_to_seconds(self, seconds: float, render_now: bool) -> None:
        if not self.is_loaded():
            return

        target = min(max(0.0, seconds), self.duration_sec if self.duration_sec > 0 else seconds)
        with self._player_lock:
            if self.player is None:
                return
            try:
                self.player.seek(int(target * 1_000_000), any_frame=False, backward=True)
                self.decode_iter = self.player.decode(video=0)
            except Exception:
                return

            frame = self._decode_next_frame_locked() if render_now else None

        if frame is not None:
            frame_time = frame.time if frame.time is not None else target
            self.current_sec = float(frame_time)
            self._last_emitted_sec = self.current_sec
            self.current_frame = int(self.current_sec * self.fps)
            rgb = frame.to_ndarray(format="rgb24")
            rgb = self._prepare_preview_rgb(rgb)
            self._render_rgb_frame(rgb)
        else:
            self.current_sec = target
            self._last_emitted_sec = self.current_sec
            self.current_frame = int(self.current_sec * self.fps)
        self._update_time_ui()

    def _update_time_ui(self) -> None:
        total_sec = int(self.duration_sec) if self.duration_sec > 0 else 0
        current_sec = int(self.current_sec)
        self.time_label.configure(text=f"{self._fmt_time(current_sec)} / {self._fmt_time(total_sec)}")

        pct = (self.current_sec / self.duration_sec * 100.0) if self.duration_sec > 0 else 0.0
        self.progress_label.configure(text=f"Progress: {self.current_frame}/{self.total_frames} ({pct:.1f}%)")

        self.ignore_seek_event = True
        self.seek_scale.set(int(self.current_sec * 1000))
        self.ignore_seek_event = False

    @staticmethod
    def _fmt_time(seconds: int) -> str:
        minutes = seconds // 60
        sec = seconds % 60
        return f"{minutes:02d}:{sec:02d}"

    def _prepare_preview_rgb(self, frame_rgb):
        max_w, max_h = 450, 420
        h, w = frame_rgb.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _render_rgb_frame(self, rgb) -> None:
        image = Image.fromarray(rgb)
        self.current_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.current_image, text="")


class VideoPlayerPanel:
    def __init__(self, root: ttk.Window, parent: Frame, title: str) -> None:
        self.root = root
        self.container = Frame(parent, relief="groove", borderwidth=1, height=350)
        self.container.pack(side=LEFT, fill=BOTH, expand=True, padx=6, pady=6)
        self.container.pack_propagate(False)  # Impede que o conteúdo redimensione o container

        self.title_label = ttk.Label(self.container, text=title, anchor="w", font=("Segoe UI", 10, "bold"))
        self.title_label.pack(fill=X, padx=8, pady=(8, 4))

        self.video_label = Label(self.container, text="No video loaded", relief="sunken")
        self.video_label.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))

        controls = Frame(self.container)
        controls.pack(fill=X, padx=8, pady=(0, 8))

        self.play_button = ttk.Button(controls, text="▶ Play", width=12, command=self.toggle_play, bootstyle="success")
        self.play_button.pack(side=LEFT)

        self.rewind_button = ttk.Button(controls, text="⏪ -5s", command=self.rewind_5s, bootstyle="secondary")
        self.rewind_button.pack(side=LEFT, padx=(8, 0))

        self.forward_button = ttk.Button(controls, text="⏩ +5s", command=self.forward_5s, bootstyle="secondary")
        self.forward_button.pack(side=LEFT, padx=(8, 0))

        self.time_label = ttk.Label(controls, text="00:00 / 00:00")
        self.time_label.pack(side=RIGHT)

        self.seek_scale = ttk.Scale(
            self.container,
            from_=0,
            to=0,
            orient=HORIZONTAL,
            command=self._on_seek,
        )
        self.seek_scale.pack(fill=X, padx=8, pady=(0, 8))

        self.progress_label = ttk.Label(self.container, text="Progress: 0/0 (0.0%)", anchor="w")
        self.progress_label.pack(fill=X, padx=8, pady=(0, 8))

        self.capture: cv2.VideoCapture | None = None
        self.video_path: str | None = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.user_can_control = True
        self.ignore_seek_event = False
        self.current_image = None
        self.target_ui_fps = 30.0
        self._capture_lock = threading.Lock()
        self._stop_worker = threading.Event()
        self._frame_queue: Queue = Queue(maxsize=2)
        self._worker_thread: threading.Thread | None = None
        self._last_ui_frame = -1

        self.root.after(30, self._tick)

    def set_title(self, text: str) -> None:
        self.title_label.configure(text=text)

    def set_controls_enabled(self, enabled: bool) -> None:
        self.user_can_control = enabled
        state = "normal" if enabled else "disabled"
        self.play_button.configure(state=state)
        self.rewind_button.configure(state=state)
        self.forward_button.configure(state=state)
        self.seek_scale.configure(state=state)

    def release(self) -> None:
        self._stop_worker.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.6)
        self._worker_thread = None
        self.is_playing = False
        with self._capture_lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

    def stop(self) -> None:
        self.is_playing = False
        self.play_button.configure(text="Play")

    def clear_display(self, text: str) -> None:
        self.video_label.configure(text=text, image="")
        self.current_image = None
        self.current_frame = 0
        self.total_frames = 0
        self._update_time_ui()

    def load_video(self, path: str) -> bool:
        self.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.clear_display("Failed to load video")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps > 0 else 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.video_path = path
        self.capture = cap

        self.ignore_seek_event = True
        self.seek_scale.configure(to=max(0, self.total_frames - 1))
        self.seek_scale.set(0)
        self.ignore_seek_event = False

        ok, frame = self.capture.read()
        if ok:
            self.current_frame = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            self._render_frame(frame)
            self._update_time_ui()
        else:
            self.clear_display("Could not read first frame")
        self.stop()
        return True

    def toggle_play(self) -> None:
        if not self.user_can_control or self.capture is None:
            return
        self.is_playing = not self.is_playing
        self.play_button.configure(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self._ensure_worker()

    def forward_5s(self) -> None:
        if not self.user_can_control or self.capture is None:
            return
        target_frame = int(self.current_frame + (self.fps * 5))
        self._seek_to_frame(target_frame)

    def rewind_5s(self) -> None:
        if not self.user_can_control or self.capture is None:
            return
        target_frame = int(self.current_frame - (self.fps * 5))
        self._seek_to_frame(target_frame)

    def set_external_frame(self, frame_bgr) -> None:
        """Render a frame from an external source (e.g., during processing).
        This does NOT stop playback - it just updates the display."""
        self._render_frame(frame_bgr)

    def set_progress(self, done: int, total: int) -> None:
        self.current_frame = max(0, done)
        self.total_frames = max(0, total)
        if self.total_frames > 0:
            self.fps = self.fps if self.fps > 0 else 30.0
        self._update_time_ui()

    def _tick(self) -> None:
        try:
            while True:
                frame_index, rgb_frame = self._frame_queue.get_nowait()
                self._render_rgb_frame(rgb_frame)
                self.current_frame = frame_index
                self._last_ui_frame = frame_index
                self._update_time_ui()
        except Empty:
            pass
        delay = max(15, int(1000 / self.target_ui_fps))
        self.root.after(delay, self._tick)

    def _on_seek(self, value: str) -> None:
        if self.ignore_seek_event or not self.user_can_control:
            return
        if self.capture is None:
            return
        self._seek_to_frame(int(float(value)))

    def _seek_to_frame(self, frame_index: int) -> None:
        if self.capture is None:
            return
        target = min(max(0, frame_index), max(0, self.total_frames - 1))
        with self._capture_lock:
            if self.capture is None:
                return
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = self.capture.read()
        if not ok:
            return
        with self._capture_lock:
            if self.capture is None:
                return
            self.current_frame = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self._render_frame(frame)
        self._update_time_ui()

    def _update_time_ui(self) -> None:
        current_seconds = int(self.current_frame / self.fps) if self.fps > 0 else 0
        total_seconds = int(self.total_frames / self.fps) if self.fps > 0 else 0
        self.time_label.configure(text=f"{self._fmt_time(current_seconds)} / {self._fmt_time(total_seconds)}")

        pct = (self.current_frame / self.total_frames * 100.0) if self.total_frames > 0 else 0.0
        self.progress_label.configure(text=f"Progress: {self.current_frame}/{self.total_frames} ({pct:.1f}%)")

        self.ignore_seek_event = True
        self.seek_scale.set(self.current_frame)
        self.ignore_seek_event = False

    @staticmethod
    def _fmt_time(seconds: int) -> str:
        minutes = seconds // 60
        sec = seconds % 60
        return f"{minutes:02d}:{sec:02d}"

    def _render_frame(self, frame_bgr) -> None:
        rgb = self._prepare_preview_rgb(frame_bgr)
        self._render_rgb_frame(rgb)

    def _prepare_preview_rgb(self, frame_bgr):
        max_w, max_h = 450, 420
        h, w = frame_bgr.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _render_rgb_frame(self, rgb) -> None:
        image = Image.fromarray(rgb)
        self.current_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.current_image, text="")

    def _ensure_worker(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_worker.clear()
        self._worker_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker_thread.start()

    def _playback_worker(self) -> None:
        while not self._stop_worker.is_set():
            if not self.is_playing:
                time.sleep(0.01)
                continue

            with self._capture_lock:
                if self.capture is None:
                    time.sleep(0.02)
                    continue

                if self.fps > self.target_ui_fps:
                    skip_count = max(0, int(self.fps / self.target_ui_fps) - 1)
                    for _ in range(skip_count):
                        self.capture.grab()

                ok, frame = self.capture.read()
                frame_index = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

            if not ok:
                self.is_playing = False
                self.root.after(0, self.stop)
                continue

            rgb = self._prepare_preview_rgb(frame)

            try:
                self._frame_queue.put_nowait((frame_index, rgb))
            except Exception:
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._frame_queue.put_nowait((frame_index, rgb))
                except Exception:
                    pass

            sleep_seconds = 1.0 / max(1.0, self.target_ui_fps)
            time.sleep(sleep_seconds)


class FacePixelApp:
    def __init__(self, root: ttk.Window) -> None:
        self.root = root
        self.root.title("FacePixel - Face Censor")
        self.root.geometry("1400x950")
        self.root.resizable(False, False)  # Tamanho fixo, não permite redimensionar

        self.video_path_var = StringVar(value="")
        self.suffix_var = StringVar(value="censored")
        self.pixelation_var = StringVar(value="15")
        self.status_var = StringVar(value="Select an MP4 video to begin.")

        self.preview_queue: Queue = Queue(maxsize=3)
        self.processing_thread: threading.Thread | None = None
        self.output_path: str | None = None
        self.pixelation_strength = 15.0
        self.detect_every_n_frames = 6  # Detectar a cada N frames
        self.cancel_flag = False

        self._build_ui()
        self._schedule_preview_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        # Header frame with better styling
        top = Frame(self.root)
        top.pack(fill=X, padx=12, pady=12)

        ttk.Label(top, text="MP4 Video:", font=("Segoe UI", 9, "bold")).pack(side=LEFT)

        self.path_entry = ttk.Entry(top, textvariable=self.video_path_var)
        self.path_entry.pack(side=LEFT, padx=8, fill=X, expand=True)

        self.pick_button = ttk.Button(top, text="📂 Browse Video", command=self.select_video, bootstyle="primary")
        self.pick_button.pack(side=RIGHT, padx=(4, 0))
        
        self.about_button = ttk.Button(top, text="ℹ️ Sobre", command=self.show_about, bootstyle="info")
        self.about_button.pack(side=RIGHT, padx=(0, 4))

        suffix_row = Frame(self.root)
        suffix_row.pack(fill=X, padx=12, pady=4)

        ttk.Label(suffix_row, text="Output suffix:", font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        self.suffix_entry = ttk.Entry(suffix_row, textvariable=self.suffix_var, width=20)
        self.suffix_entry.pack(side=LEFT, padx=8)

        self.start_button = ttk.Button(suffix_row, text="▶ Start Censoring", command=self.start_processing, bootstyle="success")
        self.start_button.pack(side=RIGHT, padx=(4, 0))

        self.cancel_button = ttk.Button(suffix_row, text="⏹ Cancel", command=self.cancel_processing, state="disabled", bootstyle="danger")
        self.cancel_button.pack(side=RIGHT, padx=(0, 4))

        # Pixelation control row with modern design
        pixelation_row = ttk.Frame(self.root)
        pixelation_row.pack(fill=X, padx=12, pady=6)

        ttk.Label(pixelation_row, text="Pixelation:", font=("Segoe UI", 10, "bold")).pack(side=LEFT, padx=(0, 8))
        
        self.pixelation_scale = ttk.Scale(
            pixelation_row,
            from_=0,
            to=30,
            orient=HORIZONTAL,
            length=250
        )
        self.pixelation_scale.pack(side=LEFT, padx=8, fill=X, expand=False)
        
        self.pixelation_label = ttk.Label(pixelation_row, text="15 pixels", font=("Segoe UI", 10, "bold"), bootstyle="info", width=12)
        self.pixelation_label.pack(side=LEFT, padx=(8, 0))
        
        # Set value and command AFTER label exists
        self.pixelation_scale.set(15)
        self.pixelation_scale.configure(command=self._on_pixelation_change)
        
        # Frame detection frequency control
        detection_row = ttk.Frame(self.root)
        detection_row.pack(fill=X, padx=12, pady=6)
        
        ttk.Label(detection_row, text="Detect every:", font=("Segoe UI", 10, "bold")).pack(side=LEFT, padx=(0, 8))
        
        self.detection_scale = ttk.Scale(
            detection_row,
            from_=1,
            to=30,
            orient=HORIZONTAL,
            length=250
        )
        self.detection_scale.pack(side=LEFT, padx=8, fill=X, expand=False)
        
        self.detection_label = ttk.Label(detection_row, text="6 frames", font=("Segoe UI", 10, "bold"), bootstyle="success", width=12)
        self.detection_label.pack(side=LEFT, padx=(8, 0))
        
        ttk.Label(detection_row, text="(lower = more accurate)", font=("Segoe UI", 8), foreground="gray").pack(side=LEFT, padx=(8, 0))
        
        # Set value and command AFTER label exists
        self.detection_scale.set(6)
        self.detection_scale.configure(command=self._on_detection_change)
        
        # GPU info row
        gpu_row = ttk.Frame(self.root)
        gpu_row.pack(fill=X, padx=12, pady=(4, 0))
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_info = f"🚀 GPU Accelerated: {gpu_name}"
            bootstyle = "success"
        else:
            gpu_info = "⚠️ Running on CPU (slower)"
            bootstyle = "warning"
        
        ttk.Label(gpu_row, text=gpu_info, font=("Segoe UI", 9, "bold"), bootstyle=bootstyle).pack(side=LEFT)

        # Preview canvas for pixelation effect
        preview_frame = Frame(self.root)
        preview_frame.pack(fill=X, padx=12, pady=6)
        
        ttk.Label(preview_frame, text="Preview:", font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        
        canvas_frame = Frame(preview_frame, relief="solid", borderwidth=2, bg="white", width=128, height=88)
        canvas_frame.pack(side=LEFT, padx=8)
        canvas_frame.pack_propagate(False)
        
        self.pixelation_preview_canvas = Canvas(
            canvas_frame,
            width=120,
            height=80,
            bg="white",
            highlightthickness=0,
        )
        self.pixelation_preview_canvas.pack(padx=4, pady=4)
        
        self._update_pixelation_preview()

        # Media players container
        media_row = Frame(self.root)
        media_row.pack(fill=BOTH, expand=True, padx=12, pady=8)

        self.source_panel = SourceVideoPlayerPanel(self.root, media_row, "Source Video")
        self.preview_panel = VideoPlayerPanel(self.root, media_row, "Censored Preview")
        self.preview_panel.set_controls_enabled(False)
        self.preview_panel.clear_display("Censored preview will appear here")

        # Status bar always visible at bottom
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, anchor="w", font=("Segoe UI", 9), relief="sunken")
        self.status_label.pack(fill=X, side="bottom", padx=12, pady=(8, 12))

    def _set_processing_state(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.pick_button.configure(state=state)
        self.start_button.configure(state=state)
        self.path_entry.configure(state=state)
        self.suffix_entry.configure(state=state)
        self.pixelation_scale.configure(state=state)
        self.detection_scale.configure(state=state)
        # Enable cancel button only during processing
        self.cancel_button.configure(state="normal" if running else "disabled")
        self.preview_panel.set_controls_enabled(not running and self.output_path is not None)

    def cancel_processing(self) -> None:
        """Cancel the ongoing video processing."""
        if not self.cancel_flag:
            self.cancel_flag = True
            self.status_var.set("Cancelling...")
            self.cancel_button.configure(state="disabled")

    def _on_pixelation_change(self, value: str) -> None:
        """Handle pixelation strength slider change."""
        try:
            self.pixelation_strength = float(value)
            self.pixelation_label.configure(text=f"{int(self.pixelation_strength)} pixels")
            self._update_pixelation_preview()  # Atualizar preview ao mover slider
        except ValueError:
            pass
    
    def _on_detection_change(self, value: str) -> None:
        """Handle detection frequency slider change."""
        try:
            self.detect_every_n_frames = int(float(value))
            self.detection_label.configure(text=f"{self.detect_every_n_frames} frames")
        except ValueError:
            pass

    def _update_pixelation_preview(self) -> None:
        """Update the pixelation preview canvas with a visual pattern that pixelates."""
        canvas = self.pixelation_preview_canvas
        canvas.delete("all")
        
        # Create a detailed image with face-like features for better preview
        img_array = np.zeros((80, 120, 3), dtype=np.uint8)
        
        # Create a gradient background
        for i in range(80):
            for j in range(120):
                r = int((j / 120) * 180 + 50)
                g = int((i / 80) * 180 + 50)
                b = int(((i + j) / 200) * 180 + 50)
                img_array[i, j] = [b, g, r]
        
        # Add some circular "facial features" for better visualization
        # Eye 1
        cv2.circle(img_array, (35, 30), 8, (255, 220, 180), -1)
        cv2.circle(img_array, (35, 30), 4, (50, 50, 50), -1)
        # Eye 2
        cv2.circle(img_array, (85, 30), 8, (255, 220, 180), -1)
        cv2.circle(img_array, (85, 30), 4, (50, 50, 50), -1)
        # Nose
        cv2.ellipse(img_array, (60, 45), (6, 10), 0, 0, 360, (200, 150, 120), -1)
        # Mouth
        cv2.ellipse(img_array, (60, 62), (15, 8), 0, 0, 180, (180, 80, 80), -1)
        
        # Apply pixelation effect based on current strength
        # Slider 0 = blocos de 2px (bem nítido)
        # Slider 30 = blocos de 50px (muito pixelado)
        block_size = max(2, int(2 + self.pixelation_strength * 1.6))
        pixel_w = max(1, 120 // block_size)
        pixel_h = max(1, 80 // block_size)
        downscaled = cv2.resize(img_array, (pixel_w, pixel_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(downscaled, (120, 80), interpolation=cv2.INTER_NEAREST)
        
        # Convert to PIL Image for display
        img_pil = Image.fromarray(pixelated)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img_pil)
        
        # Keep a reference to prevent garbage collection
        self._preview_photo = photo
        
        # Display on canvas
        canvas.create_image(60, 40, image=photo, anchor="center")

    def select_video(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            return

        file_path = filedialog.askopenfilename(
            title="Select an MP4 video",
            filetypes=[("MP4 Videos", "*.mp4")],
        )
        if not file_path:
            return

        self.video_path_var.set(file_path)
        if self.source_panel.load_video(file_path):
            self.status_var.set("Source video loaded. Click Start Censoring to process.")

    def show_about(self) -> None:
        """Exibe janela sobre o aplicativo."""
        about_window = ttk.Toplevel(self.root)
        about_window.title("Sobre - FacePixel")
        about_window.geometry("550x480")
        about_window.resizable(False, False)
        
        # Centraliza a janela
        about_window.transient(self.root)
        about_window.grab_set()
        
        # Centraliza na tela
        about_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (about_window.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (about_window.winfo_height() // 2)
        about_window.geometry(f"+{x}+{y}")
        
        # Main frame without expand to avoid compression
        main_frame = ttk.Frame(about_window, padding=20)
        main_frame.pack(fill=BOTH, expand=False)
        
        # Título
        title_label = ttk.Label(
            main_frame, 
            text="FacePixel", 
            font=("Segoe UI", 24, "bold"),
            bootstyle="primary"
        )
        title_label.pack(pady=(0, 3))
        
        # Subtítulo
        subtitle_label = ttk.Label(
            main_frame,
            text="Face Censoring Tool",
            font=("Segoe UI", 11),
            foreground="gray"
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Separator
        separator = ttk.Separator(main_frame, orient="horizontal")
        separator.pack(fill=X, pady=(0, 15))
        
        # Informações do autor
        author_section = ttk.LabelFrame(main_frame, text="Desenvolvedor")
        author_section.pack(fill=X, padx=0, pady=(0, 12))
        
        # Add padding inside the labelframe
        author_inner = ttk.Frame(author_section, padding=12)
        author_inner.pack(fill=BOTH, expand=True)
        
        author_name = ttk.Label(
            author_inner,
            text="Jean Kássio",
            font=("Segoe UI", 11, "bold")
        )
        author_name.pack(anchor="w", pady=(0, 6))
        
        # Link do GitHub (clicável)
        github_frame = ttk.Frame(author_inner)
        github_frame.pack(anchor="w", fill=X)
        
        ttk.Label(
            github_frame,
            text="GitHub:",
            font=("Segoe UI", 10)
        ).pack(side=LEFT, padx=(0, 8))
        
        github_link = ttk.Label(
            github_frame,
            text="https://github.com/jeankassio",
            font=("Segoe UI", 10, "underline"),
            foreground="#0066cc",
            cursor="hand2"
        )
        github_link.pack(side=LEFT)
        
        def open_github(event):
            import webbrowser
            webbrowser.open("https://github.com/jeankassio")
        
        github_link.bind("<Button-1>", open_github)
        
        # Features section
        features_section = ttk.LabelFrame(main_frame, text="Recursos")
        features_section.pack(fill=BOTH, expand=False, pady=(0, 12))
        
        # Add padding inside the labelframe
        features_inner = ttk.Frame(features_section, padding=12)
        features_inner.pack(fill=BOTH, expand=True)
        
        features = [
            "✓ Detecção GPU acelerada (InsightFace RetinaFace)",
            "✓ Pixelação e criptografia de rostos",
            "✓ Interface intuitiva e responsiva",
            "✓ Processamento em tempo real",
            "✓ Suporte a vídeos MP4"
        ]
        
        for feature in features:
            feature_label = ttk.Label(
                features_inner,
                text=feature,
                font=("Segoe UI", 9),
                wraplength=420
            )
            feature_label.pack(anchor="w", pady=2)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=(5, 0))
        
        # Botão fechar
        close_button = ttk.Button(
            button_frame,
            text="Fechar",
            command=about_window.destroy,
            bootstyle="secondary",
            width=25
        )
        close_button.pack(pady=5)

    def start_processing(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            return

        input_path = self.video_path_var.get().strip()
        if not input_path:
            messagebox.showerror("Error", "Please select an MP4 video first.")
            return

        self._start_processing(input_path)

    def _start_processing(self, input_path: str) -> None:
        if not Path(input_path).exists():
            messagebox.showerror("Error", "The selected file does not exist.")
            return

        if not self.source_panel.is_loaded():
            if not self.source_panel.load_video(input_path):
                messagebox.showerror("Error", "Could not load source video.")
                return

        output_path = build_output_path(input_path, self.suffix_var.get())
        self.output_path = None
        self.status_var.set(f"Processing: {Path(input_path).name}")
        self._set_processing_state(True)
        self.preview_panel.stop()
        self.preview_panel.set_title("Censored Preview (Processing)")
        self.preview_panel.clear_display("Processing preview...")

        def on_preview(frame_bgr):
            try:
                # Non-blocking put - skip frame if preview queue is full
                # This prevents the processing thread from being blocked by slow UI updates
                self.preview_queue.put_nowait(frame_bgr)
            except Exception:
                # Queue full - drop frame, next one will come soon
                pass

        def on_progress(done: int, total: int):
            if total > 0:
                pct = (done / total) * 100
                self.root.after(
                    0,
                    lambda d=done, t=total, p=pct: (
                        self.status_var.set(f"Processing... {d}/{t} ({p:.1f}%)"),
                        self.preview_panel.set_progress(d, t),
                    ),
                )
            else:
                self.root.after(
                    0,
                    lambda d=done: (
                        self.status_var.set(f"Processing... {d} frames"),
                        self.preview_panel.set_progress(d, d),
                    ),
                )

        def worker() -> None:
            try:
                result_path = censor_video(
                    input_path=input_path,
                    output_path=output_path,
                    on_preview=on_preview,
                    on_progress=on_progress,
                    pixelation_strength=self.pixelation_strength,
                    detect_every_n_frames=self.detect_every_n_frames,
                    cancel_check=lambda: self.cancel_flag,
                )
                # Check if cancelled after processing completes
                if self.cancel_flag:
                    self.root.after(0, lambda p=output_path: self._on_cancelled(p))
                else:
                    self.root.after(
                        0,
                        lambda: self._on_success(result_path),
                    )
            except Exception as exc:
                error_message = str(exc)
                self.root.after(0, lambda msg=error_message: self._on_error(msg))

        self.processing_thread = threading.Thread(target=worker, daemon=True)
        self.processing_thread.start()

    def _schedule_preview_update(self) -> None:
        try:
            frame_bgr = self.preview_queue.get_nowait()
            self.preview_panel.set_external_frame(frame_bgr)
        except Empty:
            pass
        self.root.after(30, self._schedule_preview_update)

    def _on_success(self, output_path: str) -> None:
        self.output_path = output_path
        self._set_processing_state(False)
        self.status_var.set(f"Done! Output file created: {output_path}")
        self.preview_panel.set_title("Censored Output")
        if self.preview_panel.load_video(output_path):
            self.preview_panel.set_controls_enabled(True)
        messagebox.showinfo("Completed", f"Censored video saved to:\n{output_path}")

    def _on_error(self, message: str) -> None:
        self._set_processing_state(False)
        self.status_var.set("Failed to process video.")
        self.preview_panel.set_title("Censored Preview")
        messagebox.showerror("Error", message)

    def _on_cancelled(self, output_path: str) -> None:
        """Handle cancelled processing - delete partial file and reset state."""
        self._set_processing_state(False)
        self.cancel_flag = False
        
        # Try to delete the partial output file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"[DEBUG] Deleted partial file: {output_path}")
            except Exception as e:
                print(f"[WARNING] Failed to delete partial file: {e}")
        
        self.status_var.set("Processing cancelled.")
        self.preview_panel.set_title("Censored Preview")
        messagebox.showinfo("Cancelled", "Processing was cancelled and partial file was deleted.")

    def _on_close(self) -> None:
        """Close the application properly and cleanup resources."""
        try:
            # Stop any ongoing processing
            if self.processing_thread and self.processing_thread.is_alive():
                self.cancel_flag = True
                self.processing_thread.join(timeout=2.0)
            
            # Release all video resources
            self.source_panel.release()
            self.preview_panel.release()
            
            # Destroy window
            self.root.destroy()
            
            # Force exit to ensure all background threads are killed
            import sys
            sys.exit(0)
        except Exception as e:
            print(f"[WARNING] Error during close: {e}")
            import sys
            sys.exit(0)


def main() -> None:
    print("[INFO] FacePixel starting...")
    
    # Ensure DeepFace models are available before starting the app
    if not ensure_deepface_models():
        print("[WARNING] DeepFace models could not be loaded")
        print("[WARNING] The app will still run, but face detection may fail")
        print("[WARNING] Please ensure you have internet connection for first run")
    
    root = ttk.Window(themename="darkly")
    app = FacePixelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()