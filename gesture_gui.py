import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from Levenshtein import distance as levenshtein_distance

# ============================== CONFIG ==============================
GESTURES = ["wave", "peace sign", "fist"]
PASSWORD_ORDER = [0, 1, 2]
T = 30

SEQUENCE_FILE = "latest_sequence.npy"
RESULT_FILE = "result.json"

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

# ============================== GUI ==============================
class GestureApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Gesture Password Access Control")
        self.geometry("900x700")

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

        self.sequence_label = tk.Label(
            self,
            text="Current sequence: []",
            font=("Arial", 12),
        )
        self.sequence_label.pack(pady=5)

        self.result_label = tk.Label(
            self,
            text="Model result: waiting",
            font=("Arial", 12),
        )
        self.result_label.pack(pady=5)

        self.capture_btn = tk.Button(
            self,
            text="Capture 2-second gesture",
            command=self.capture_sequence,
            font=("Arial", 12),
        )
        self.capture_btn.pack(pady=10)

        self.reset_btn = tk.Button(
            self,
            text="Reset display",
            command=self.reset_display,
            font=("Arial", 12),
        )
        self.reset_btn.pack(pady=5)

        self.close_btn = tk.Button(
            self,
            text="Quit",
            command=self.destroy,
            font=("Arial", 12),
        )
        self.close_btn.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.current_sequence = []
        self.last_result_mtime = None

        self.update_webcam()
        self.poll_result_file()

    def reset_display(self):
        self.current_sequence = []
        self.status_label.config(text="STATUS: RESET", bg="yellow", fg="black")
        self.sequence_label.config(text="Current sequence: []")
        self.result_label.config(text="Model result: waiting")

    def capture_sequence(self):
        self.status_label.config(text="CAPTURING...", bg="cyan", fg="black")
        self.update_idletasks()

        frames = []
        start_time = time.time()

        while time.time() - start_time < 2.0:
            ok, frame = self.cap.read()
            if not ok:
                continue
            lm = extract_3d_landmarks(frame)
            if lm is not None:
                frames.append(lm)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(img)
            self.webcam_label.config(image=img)
            self.webcam_label.image = img
            self.update()

        if len(frames) == 0:
            self.status_label.config(text="NO HAND DETECTED", bg="orange", fg="black")
            return

        seq_data = np.array(frames, dtype="float32")
        if len(seq_data) >= T:
            seq_data = seq_data[:T]
        else:
            pad = np.repeat(seq_data[-1:], T - len(seq_data), axis=0)
            seq_data = np.concatenate([seq_data, pad], axis=0)

        np.save(SEQUENCE_FILE, seq_data)
        self.status_label.config(text="SEQUENCE SAVED", bg="lightblue", fg="black")
        self.result_label.config(text="Model result: waiting for inference...")

    def poll_result_file(self):
        if os.path.exists(RESULT_FILE):
            mtime = os.path.getmtime(RESULT_FILE)
            if self.last_result_mtime is None or mtime > self.last_result_mtime:
                self.last_result_mtime = mtime
                try:
                    with open(RESULT_FILE, "r") as f:
                        result = json.load(f)

                    decision = result.get("decision", "UNKNOWN")
                    predicted_id = result.get("predicted_id", -1)
                    predicted_label = result.get("predicted_label", "unknown")
                    current_sequence = result.get("current_sequence", [])

                    self.sequence_label.config(
                        text=f"Current sequence: {current_sequence}"
                    )
                    self.result_label.config(
                        text=f"Model result: {predicted_label} (ID={predicted_id})"
                    )

                    if decision == "ACCESS GRANTED":
                        self.status_label.config(text=decision, bg="green", fg="white")
                    elif decision == "ACCESS DENIED":
                        self.status_label.config(text=decision, bg="red", fg="white")
                    else:
                        self.status_label.config(text=decision, bg="yellow", fg="black")
                except Exception as e:
                    self.result_label.config(text=f"Result read error: {e}")

        self.after(500, self.poll_result_file)

    def update_webcam(self):
        ok, frame = self.cap.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(img)
            self.webcam_label.config(image=img)
            self.webcam_label.image = img

        self.after(20, self.update_webcam)

    def destroy(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        super().destroy()

if __name__ == "__main__":
    app = GestureApp()
    app.mainloop()