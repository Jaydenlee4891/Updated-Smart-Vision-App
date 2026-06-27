"""
Smart Vision App — app.py
Real-time object detection + TTS for visually-impaired users.
Built with Kivy · TensorFlow Lite · YOLOv8
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from plyer import tts

from config import (
    CONF_THRESHOLD,
    IMG_SIZE,
    MODEL_PATH,
    NMS_THRESHOLD,
    PRIORITY_CLASSES,
    PRIORITY_COOLDOWN,
    TARGET_FPS,
    TTS_COOLDOWN,
)
from names import ALERT_PHRASES, NAMES
from util import apply_nms, draw_detections, parse_yolo_output, preprocess


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_interpreter(model_path: str) -> tf.lite.Interpreter:
    """Load the TFLite interpreter; raise a clear error if the file is absent."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'\n"
            "Place 'best_float32.tflite' inside the model/ directory."
        )
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


# ---------------------------------------------------------------------------
# Text-to-Speech announcer
# ---------------------------------------------------------------------------

class Announcer:
    """
    Thread-safe TTS wrapper with per-class cooldown.
    Priority classes (crossroads, red lights, people ahead) use a shorter
    cooldown so they are re-announced quickly if still visible.
    """

    def __init__(self) -> None:
        self._last_spoken: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def speak(self, label: str) -> bool:
        """
        Announce *label* if enough time has passed since the last announcement.
        Returns True when speech is triggered.
        """
        phrase   = ALERT_PHRASES.get(label, label)
        cooldown = PRIORITY_COOLDOWN if label in PRIORITY_CLASSES else TTS_COOLDOWN

        with self._lock:
            now = time.monotonic()
            if now - self._last_spoken[label] < cooldown:
                return False
            self._last_spoken[label] = now

        # Fire-and-forget on a daemon thread so TTS never blocks the UI
        threading.Thread(target=self._say, args=(phrase,), daemon=True).start()
        return True

    @staticmethod
    def _say(text: str) -> None:
        try:
            tts.speak(text)
        except Exception:
            # TTS unavailable on desktop (normal during development) — silent
            pass


# ---------------------------------------------------------------------------
# Background inference worker
# ---------------------------------------------------------------------------

class InferenceWorker:
    """
    Runs TFLite inference on a dedicated background thread so the Kivy UI
    thread is never blocked by model invocation.

    Uses a latest-frame policy: if a new frame arrives before the previous
    inference finishes, the old frame is discarded.
    """

    def __init__(self, interpreter: tf.lite.Interpreter, announcer: Announcer) -> None:
        self._interpreter     = interpreter
        self._input_details   = interpreter.get_input_details()
        self._output_details  = interpreter.get_output_details()
        self._announcer       = announcer

        self._lock             = threading.Lock()
        self._pending_frame:   np.ndarray | None = None
        self._latest_detections: list = []
        self._running          = False
        self._thread           = threading.Thread(target=self._loop, daemon=True)

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def submit(self, frame: np.ndarray) -> None:
        """Enqueue the latest camera frame (older pending frame is dropped)."""
        with self._lock:
            self._pending_frame = frame.copy()

    def get_detections(self) -> list:
        """Return the most recently completed detection list (thread-safe)."""
        with self._lock:
            return list(self._latest_detections)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            detections = self._infer(frame)

            with self._lock:
                self._latest_detections = detections

            # Trigger TTS for each unique detected class
            seen: set[str] = set()
            for *_, cid in detections:
                label = NAMES.get(int(cid), "")
                if label and label not in seen:
                    self._announcer.speak(label)
                    seen.add(label)

    def _infer(self, frame: np.ndarray) -> list:
        tensor, orig_shape = preprocess(frame, (IMG_SIZE, IMG_SIZE))
        self._interpreter.set_tensor(self._input_details[0]['index'], tensor)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_details[0]['index'])
        detections = parse_yolo_output(raw, CONF_THRESHOLD, orig_shape, IMG_SIZE)
        return apply_nms(detections, NMS_THRESHOLD)


# ---------------------------------------------------------------------------
# Kivy application
# ---------------------------------------------------------------------------

class SmartVisionApp(App):

    title = "Smart Vision"

    # ------------------------------------------------------------------
    def build(self):
        self.announcer: Announcer | None = None
        self.worker:    InferenceWorker | None = None
        self.cap:       cv2.VideoCapture | None = None
        self._running   = False

        # ── Load model ─────────────────────────────────────────────────
        try:
            interpreter = load_interpreter(MODEL_PATH)
        except FileNotFoundError as exc:
            return self._error_screen(str(exc))

        self.announcer = Announcer()
        self.worker    = InferenceWorker(interpreter, self.announcer)
        self.worker.start()

        # ── Root layout ────────────────────────────────────────────────
        root = BoxLayout(orientation='vertical', padding=8, spacing=6)

        # Status bar
        self.status_lbl = Label(
            text="Smart Vision  ·  Ready",
            size_hint_y=None, height=30,
            font_size="14sp",
            color=(0.55, 1.0, 0.65, 1),
        )
        root.add_widget(self.status_lbl)

        # Live camera feed
        self.img_widget = Image(allow_stretch=True)
        root.add_widget(self.img_widget)

        # Detection summary line
        self.detection_lbl = Label(
            text="",
            size_hint_y=None, height=28,
            font_size="13sp",
            color=(1, 1, 0.55, 1),
        )
        root.add_widget(self.detection_lbl)

        # Buttons
        btn_row = BoxLayout(size_hint_y=None, height=58, spacing=8)
        start_btn = Button(
            text="▶  Start",
            background_color=(0.1, 0.65, 0.3, 1),
            on_press=self.start_camera,
        )
        stop_btn = Button(
            text="■  Stop",
            background_color=(0.75, 0.15, 0.15, 1),
            on_press=self.stop_camera,
        )
        btn_row.add_widget(start_btn)
        btn_row.add_widget(stop_btn)
        root.add_widget(btn_row)

        Clock.schedule_interval(self._update, 1.0 / TARGET_FPS)
        return root

    # ------------------------------------------------------------------
    def start_camera(self, *_) -> None:
        if self._running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_lbl.text = "⚠  Camera not found — check permissions"
            return
        self._running = True
        self.status_lbl.text = "Smart Vision  ·  Running"

    def stop_camera(self, *_) -> None:
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_lbl.text = "Smart Vision  ·  Stopped"
        self.detection_lbl.text = ""

    # ------------------------------------------------------------------
    def _update(self, dt: float) -> None:
        """Called by Kivy's clock every frame to refresh the UI."""
        if not self._running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 0)

        # Push frame to background inference thread
        if self.worker:
            self.worker.submit(frame)

        # Annotate with latest results (may lag one frame behind — fine)
        detections = self.worker.get_detections() if self.worker else []
        if detections:
            frame = draw_detections(frame.copy(), detections, NAMES)
            unique_labels = list(dict.fromkeys(
                NAMES.get(int(d[5]), "") for d in detections
            ))
            self.detection_lbl.text = "  ·  ".join(filter(None, unique_labels))
        else:
            self.detection_lbl.text = ""

        # Render to Kivy texture
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        texture = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture

    # ------------------------------------------------------------------
    def on_stop(self) -> None:
        """Clean up when the app is closed."""
        self.stop_camera()
        if self.worker:
            self.worker.stop()

    # ------------------------------------------------------------------
    @staticmethod
    def _error_screen(message: str):
        layout = BoxLayout(orientation='vertical', padding=24)
        layout.add_widget(Label(
            text=f"[b]Could not start Smart Vision[/b]\n\n{message}",
            markup=True,
            halign="center",
            valign="middle",
            color=(1.0, 0.4, 0.4, 1),
        ))
        return layout


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SmartVisionApp().run()
