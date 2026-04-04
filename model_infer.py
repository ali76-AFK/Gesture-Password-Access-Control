import numpy as np
import json
import os
import time
from tensorflow.keras.models import load_model
from Levenshtein import distance as levenshtein_distance

# ============================== CONFIG ==============================
GESTURES = ["wave", "peace sign", "fist"]
PASSWORD_ORDER = [0, 1, 2]
T = 30

SEQUENCE_FILE = "latest_sequence.npy"
RESULT_FILE = "result.json"

POLL_INTERVAL = 0.5

# ============================== MODEL ==============================
model = load_model("gesture_classifier.h5", compile=False)

# ============================== HELPERS ==============================
def levenshtein_tolerance_match(predicted_seq, password_order, tolerance_frac=0.1):
    n = len(password_order)
    max_allowed_edits = max(1, int(n * tolerance_frac))
    d = levenshtein_distance(predicted_seq, password_order)
    return d <= max_allowed_edits

def load_sequence(file_path):
    seq = np.load(file_path)
    if len(seq) >= T:
        seq = seq[:T]
    else:
        pad = np.repeat(seq[-1:], T - len(seq), axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    return seq[None, :, :]

# ============================== MAIN LOOP ==============================
def main():
    print("Model inference service started.")
    print(f"Watching: {SEQUENCE_FILE}")

    last_mtime = None
    current_sequence = []

    while True:
        if os.path.exists(SEQUENCE_FILE):
            mtime = os.path.getmtime(SEQUENCE_FILE)

            if last_mtime is None or mtime > last_mtime:
                last_mtime = mtime

                try:
                    seq_input = load_sequence(SEQUENCE_FILE)
                    pred = model(seq_input, training=False).numpy()
                    class_id = int(np.argmax(pred, axis=-1)[0])
                    predicted_label = GESTURES[class_id]

                    current_sequence.append(class_id)

                    if len(current_sequence) >= len(PASSWORD_ORDER):
                        if levenshtein_tolerance_match(
                            current_sequence, PASSWORD_ORDER, tolerance_frac=0.1
                        ):
                            decision = "ACCESS GRANTED"
                        else:
                            decision = "ACCESS DENIED"
                        sequence_to_write = current_sequence.copy()
                        current_sequence = []
                    else:
                        decision = "WAITING"
                        sequence_to_write = current_sequence.copy()

                    result = {
                        "timestamp": time.time(),
                        "predicted_id": class_id,
                        "predicted_label": predicted_label,
                        "current_sequence": sequence_to_write,
                        "decision": decision,
                    }

                    with open(RESULT_FILE, "w") as f:
                        json.dump(result, f, indent=2)

                    print(result)

                except Exception as e:
                    error_result = {
                        "timestamp": time.time(),
                        "predicted_id": -1,
                        "predicted_label": "error",
                        "current_sequence": current_sequence,
                        "decision": f"ERROR: {e}",
                    }
                    with open(RESULT_FILE, "w") as f:
                        json.dump(error_result, f, indent=2)
                    print(error_result)

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()