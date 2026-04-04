import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Configuration
GESTURES = ["wave", "peace sign", "fist"]
N_CLASSES = len(GESTURES)
FRAMES_PER_GESTURE = 30   # 2 seconds at ~15 fps

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def extract_3d_landmarks(frame):
    # 3D landmarks (world coordinates) => 63‑dim vector
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if not results.multi_hand_world_landmarks:
        return None
    hand_land = results.multi_hand_world_landmarks[0]
    vec = []
    for lm in hand_land.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype="float32")

def record_gesture_sequence(gesture_name, gesture_label, n_samples=10, duration_seconds=2.0):
    print(f"RECORDING GESTURE: {gesture_name}")
    print("Press SPACE when you're ready to start.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera!")
        return [], []

    sequences = []
    labels = []

    for i in range(n_samples):
        input("Press Enter when you're ready for sample %d / %d: " % (i+1, n_samples))
        print("Recording 2 seconds NOW... do the gesture!")
        frames = []
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            success, frame = cap.read()
            if not success:
                continue
            lm = extract_3d_landmarks(frame)
            if lm is not None:
                frames.append(lm)
            # Visual feedback
            cv2.imshow("Webcam (record)", frame)
            if cv2.waitKey(1) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return [], []
        print("Done.\n")
        cap.release()
        cap = cv2.VideoCapture(0)  # re‑open for next sample

        seq = np.array(frames, dtype="float32")  # shape (T, 63)
        if len(seq) == 0:
            print("Warning: no frames recorded for this sample; skipping.")
            continue
        sequences.append(seq)
        labels.append(gesture_label)

    cv2.destroyAllWindows()
    return sequences, labels

# Record all gestures
X_list = []
y_list = []

for label_idx, gesture_name in enumerate(GESTURES):
    seqs, lbls = record_gesture_sequence(
        gesture_name, label_idx, n_samples=10, duration_seconds=2.0
    )
    X_list.extend(seqs)
    y_list.extend(lbls)

# Save to disk
X_train = np.array(X_list, dtype="object")  # list of variable‑length sequences
y_train = np.array(y_list, dtype="int32")

np.savez_compressed("gesture_data.npz", X=X_train, y=y_train)
print("Saved gesture_data.npz (X.shape = X_train, y.shape = y_train).")