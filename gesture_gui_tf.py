import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from Levenshtein import distance as levenshtein_distance
import time

# ============================== CONFIG ==============================
GESTURES = ["wave", "peace sign", "fist"]
PASSWORD_ORDER = [0, 1, 2]
N_CLASSES = len(GESTURES)
T = 30
FPS = 15

# ============================== MODEL ==============================
model = load_model("gesture_classifier.h5", compile=False)

# ============================== MediaPipe Hands ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def extract_3d_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if not results.multi_hand_world_landmarks:
        return None
    hand_land = results.multi_hand_world_landmarks[0]
    vec = []
    for lm in hand_land.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype="float32")

def levenshtein_tolerance_match(predicted_seq, password_order, tolerance_frac=0.1):
    n = len(password_order)
    max_allowed_edits = max(1, int(n * tolerance_frac))
    d = levenshtein_distance(predicted_seq, password_order)
    return d <= max_allowed_edits

# ============================== GUI ==============================
class GestureApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Gesture Password Access Control")
        self.geometry("800x600")

        self.webcam_label = tk.Label(self, text="Webcam", bg="black", fg="white")
        self.webcam_label.pack(pady=10)

        self.status_label = tk.Label(
            self,
            text="STATUS: WAITING",
            bg="yellow",
            fg="black",
            font=("Arial", 16, "bold"),
        )
        self.status_label.pack(pady=10)

        self.attempt_label = tk.Label(self, text="Attempts: 0 / 3", font=("Arial", 12))
        self.attempt_label.pack(pady=5)

        self.reset_btn = tk.Button(self, text="Reset sequence", command=self.reset_sequence)
        self.reset_btn.pack(pady=5)

        self.close_btn = tk.Button(self, text="Quit", command=self.destroy)
        self.close_btn.pack(pady=10)

        self.cap = cv2.VideoCapture(0)

        self.current_sequence = []
        self.attempt = 0
        self.lockout_until = 0
        self.MAX_ATTEMPTS = 3
        self.LOCKOUT_SECONDS = 15
        self.last_gesture_time = time.time()

        self.update_webcam()

    def reset_sequence(self):
        self.current_sequence = []
        self.attempt = 0
        self.status_label.config(text="STATUS: RESETTING", bg="yellow")
        self.attempt_label.config(text=f"Attempts: {self.attempt} / 3")

    def update_webcam(self):
        if time.time() < self.lockout_until:
            self.status_label.config(text="STATUS: LOCKOUT (wait)", bg="orange")
            self.after(100, self.update_webcam)
            return

        success, frame = self.cap.read()
        if not success:
            self.after(20, self.update_webcam)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(img)
        self.webcam_label.config(image=img)
        self.webcam_label.image = img

        if time.time() - self.last_gesture_time >= 2.0:
            frames = []
            start_time = time.time()

            while time.time() - start_time < 2.0:
                ok, f = self.cap.read()
                if not ok:
                    continue
                lm = extract_3d_landmarks(f)
                if lm is not None:
                    frames.append(lm)

            if len(frames) > 0:
                seq_data = np.array(frames, dtype="float32")
                if len(seq_data) >= T:
                    seq_data = seq_data[:T]
                else:
                    pad = np.repeat(seq_data[-1:], T - len(seq_data), axis=0)
                    seq_data = np.concatenate([seq_data, pad], axis=0)
                seq_data = seq_data[None, :, :]  # (1, T, 63)

                pred = model(seq_data, training=False).numpy()
                class_id = int(np.argmax(pred, axis=-1)[0])
                print("Detected:", GESTURES[class_id], "(ID =", class_id, ")")
                self.current_sequence.append(class_id)

                if len(self.current_sequence) >= len(PASSWORD_ORDER):
                    if levenshtein_tolerance_match(
                        self.current_sequence, PASSWORD_ORDER, tolerance_frac=0.1
                    ):
                        self.status_label.config(text="ACCESS GRANTED", bg="green", fg="white")
                        self.attempt = 0
                    else:
                        self.status_label.config(text="ACCESS DENIED", bg="red", fg="white")
                        self.attempt += 1
                        if self.attempt >= self.MAX_ATTEMPTS:
                            self.status_label.config(
                                text=f"LOCKOUT (wait {self.LOCKOUT_SECONDS}s)",
                                bg="orange",
                                fg="black",
                            )
                            self.lockout_until = time.time() + self.LOCKOUT_SECONDS
                    self.current_sequence = []

            self.last_gesture_time = time.time()

        self.attempt_label.config(text=f"Attempts: {self.attempt} / 3")
        self.after(20, self.update_webcam)

    def destroy(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        super().destroy()

if __name__ == "__main__":
    app = GestureApp()
    app.mainloop()