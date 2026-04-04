import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import requests
import time

GESTURES = ["wave", "peace sign", "fist"]
PASSWORD_ORDER = [0, 1, 2]
T = 30
FPS = 15

API_URL = "https://elaine-mitral-gabriel.ngrok-free.dev"
SEQUENCE_ENDPOINT = f"{API_URL}/sequence"
RESULT_ENDPOINT = f"{API_URL}/latest_result"

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

class GestureApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Gesture Password Access Control (HTTP API)")
        self.geometry("1000x700")

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
            text="API result: waiting",
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
            text="Reset sequence",
            command=self.reset_sequence,
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
        self.last_api_poll = 0

        self.update_webcam()
        self.poll_api()

    def reset_sequence(self):
        self.current_sequence = []
        self.sequence_label.config(text="Current sequence: []")
        self.status_label.config(text="STATUS: RESETTING", bg="yellow", fg="black")
        self.result_label.config(text="API result: waiting")

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
            self.status_label.config(
                text="NO HAND DETECTED", bg="orange", fg="black"
            )
            return

        seq_data = np.array(frames, dtype="float32")
        if len(seq_data) >= T:
            seq_data = seq_data[:T]
        else:
            pad = np.repeat(seq_data[-1:], T - len(seq_data), axis=0)
            seq_data = np.concatenate([seq_data, pad], axis=0)

        payload = {
            "sequence": self.current_sequence,
            "landmarks": seq_data.tolist()
        }

        try:
            r = requests.post(SEQUENCE_ENDPOINT, json=payload, timeout=10)
            if r.ok:
                result = r.json()
                print("API response:", result)

                predicted_id = result.get("predicted_id", -1)
                if predicted_id >= 0 and len(self.current_sequence) < len(PASSWORD_ORDER):
                    self.current_sequence.append(predicted_id)

                self.sequence_label.config(
                    text=f"Current sequence: {self.current_sequence}"
                )

                if len(self.current_sequence) == len(PASSWORD_ORDER):
                    if self.current_sequence == PASSWORD_ORDER:
                        self.status_label.config(
                            text="ACCESS GRANTED", bg="green", fg="white"
                        )
                    else:
                        self.status_label.config(
                            text="ACCESS DENIED", bg="red", fg="white"
                        )
                else:
                    self.status_label.config(
                        text="WAITING", bg="yellow", fg="black"
                    )

                self.update_labels_from_result(result)
            else:
                self.result_label.config(
                    text=f"API error {r.status_code}: {r.text}"
                )
        except Exception as e:
            self.result_label.config(text=f"API request error: {e}")

    def update_labels_from_result(self, result):
        decision = result.get("decision", "UNKNOWN")
        predicted_id = result.get("predicted_id", -1)
        predicted_label = result.get("predicted_label", "unknown")

        self.result_label.config(
            text=f"API result: {predicted_label} (ID={predicted_id})"
        )

    def poll_api(self):
        now = time.time()
        if now - self.last_api_poll > 1.0:
            self.last_api_poll = now
            try:
                r = requests.get(RESULT_ENDPOINT, timeout=5)
                if r.ok:
                    result = r.json()
                    print("API response:", result)
                    self.update_labels_from_result(result)
            except Exception:
                pass
        self.after(100, self.poll_api)

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