import numpy as np
from tensorflow.keras.models import load_model
from Levenshtein import distance as levenshtein_distance

# 1. Load model
model = load_model("gesture_classifier.h5")

# 2. Load recorded data
data = np.load("gesture_data.npz", allow_pickle=True)
X_train_seq = data["X"]
y_train = data["y"]

# 3. Pick a test sample
T = 30
seq = X_train_seq[0].copy()
if len(seq) >= T:
    seq = seq[:T]
else:
    pad = np.repeat(seq[-1:], T - len(seq), axis=0)
    seq = np.concatenate([seq, pad], axis=0)

# 4. Inference
seq = seq[None, :, :]  # (1, T, 63)
pred = model(seq, training=False).numpy()
class_id = np.argmax(pred, axis=-1)[0]

GESTURES = ["wave", "peace sign", "fist"]
PASSWORD_ORDER = [0, 1, 2]

print("Detected gesture ID:", class_id)
print("Gesture label:", GESTURES[class_id])

# 5. Sequence matching test
EXAMPLE_SEQUENCE = [0, 1, 2]

def levenshtein_tolerance_match(predicted_seq, password_order, tolerance_frac=0.1):
    n = len(password_order)
    max_allowed_edits = max(1, int(n * tolerance_frac))
    d = levenshtein_distance(predicted_seq, password_order)
    return d <= max_allowed_edits

match_ok = levenshtein_tolerance_match(EXAMPLE_SEQUENCE, PASSWORD_ORDER, tolerance_frac=0.1)
print("\nSequence‑matching test:")
print("Given sequence:", EXAMPLE_SEQUENCE)
print("Password order:", PASSWORD_ORDER)
print("Levenshtein distance:", levenshtein_distance(EXAMPLE_SEQUENCE, PASSWORD_ORDER))
print("10%‑tolerant match OK?", match_ok)
print("ACCESS GRANTED" if match_ok else "ACCESS DENIED")