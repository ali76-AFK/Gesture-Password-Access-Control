# test_local.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from Levenshtein import distance as levenshtein_distance

# 1. Load model
model = load_model("gesture_classifier.h5")  # or .keras if you saved that

# 2. Gesture config
GESTURES = ["wave", "peace sign", "fist"]
PASSWORD_ORDER = [0, 1, 2]  # wave → peace sign → fist
T = 30  # 30 frames (about 2 seconds at 15 fps)

# 3. MediaPipe Hands
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

# 4. Webcam loop (local inference only)
def run_local_inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    attempt = 0
    LOCKOUT_SECONDS = 15
    lockout_until = 0
    current_sequence = []

    print("Starting local gesture password system...")
    print("Password: wave → peace sign → fist")
    print("Press 'q' to quit.")

    while cap.isOpened():
        if time.time() < lockout_until:
            print("LOCKOUT: wait...")
            time.sleep(0.1)
            continue

        # Record 2 seconds of frames
        frames = []
        start_time = time.time()
        while time.time() - start_time < 2.0:
            success, frame = cap.read()
            if not success:
                continue
            lm = extract_3d_landmarks(frame)
            if lm is not None:
                frames.append(lm)
            cv2.imshow("Gesture Password (local)", frame)
            if cv2.waitKey(1) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        if len(frames) == 0:
            continue

        # Pad/truncate to T
        seq_data = np.array(frames, dtype="float32")
        if len(seq_data) >= T:
            seq_data = seq_data[:T]
        else:
            pad = np.repeat(seq_data[-1:], T - len(seq_data), axis=0)
            seq_data = np.concatenate([seq_data, pad], axis=0)
        seq_data = seq_data[None, :, :]  # (1, T, 63)

        # Inference
        pred = model(seq_data, training=False).numpy()
        class_id = np.argmax(pred, axis=-1)[0]

        print("Detected:", GESTURES[class_id], "(ID =", class_id, ")")

        current_sequence.append(int(class_id))

        if len(current_sequence) >= len(PASSWORD_ORDER):
            if levenshtein_tolerance_match(current_sequence, PASSWORD_ORDER, tolerance_frac=0.1):
                print("✅ ACCESS GRANTED")
            else:
                print("❌ ACCESS DENIED")
                attempt += 1
                if attempt >= 3:
                    print("Too many failed attempts. Lockout for", LOCKOUT_SECONDS, "s.")
                    lockout_until = time.time() + LOCKOUT_SECONDS
            current_sequence = []

    cap.release()
    cv2.destroyAllWindows()

# 5. Run
if __name__ == "__main__":
    import time
    run_local_inference()