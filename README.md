# Gesture Password Access Control

A hand-gesture access control project that combines a local desktop GUI, MediaPipe hand landmark extraction, a GRU gesture classifier, and a Flask API exposed through ngrok.

The system works like a gesture-based password lock:
- You perform a gesture in front of the webcam.
- The desktop app extracts 3D hand landmarks with MediaPipe.
- The landmark sequence is sent to a Flask API.
- The GRU model predicts the gesture class.
- The GUI appends the predicted gesture ID to the current password sequence.
- Access is granted only when the predicted sequence exactly matches the configured password order.

## Features

- Desktop GUI using Tkinter.
- Real-time webcam preview with OpenCV.
- 3D hand landmark extraction using MediaPipe Hands.
- GRU-based gesture classification using TensorFlow/Keras.
- HTTP-based inference pipeline using Flask.
- Public API sharing with ngrok.
- Exact sequence matching for secure gesture-password verification.
- Resettable gesture sequence from the GUI.

## Project Structure

```text
GesturePasswordAccess/
├── gesture_gui_http.py      # Desktop GUI, webcam capture, landmark extraction, HTTP client
├── model_api.py             # Flask inference API and ngrok tunnel launcher
├── gesture_classifier.h5    # Trained GRU model
├── result.json              # Last server prediction result (generated at runtime)
├── requirements.txt         # Python dependencies
├── .gitignore               # Files/folders ignored by Git
└── README.md                # Project documentation
```

## How It Works

### 1. GUI side
The local GUI captures a 2-second gesture clip from your webcam. For each frame, MediaPipe extracts 21 hand landmarks with x, y, and z coordinates, producing 63 values per frame. MediaPipe Hands detects 21 landmarks per hand, which is why each frame becomes a 63-element feature vector. [1][2]

The GUI collects these frame vectors into a sequence of length 30. If fewer than 30 frames are detected, the last frame is repeated until the sequence length reaches 30.

### 2. API side
The Flask API receives:
- the current local gesture sequence,
- the captured landmark tensor.

The API loads the GRU model from `gesture_classifier.h5`, predicts one of the configured gesture classes, and returns the predicted class ID and label.

### 3. Password logic
The GUI appends each returned prediction to `current_sequence`. Access is granted only when the final sequence exactly equals:

```python
PASSWORD_ORDER = [0, 1, 2]
```

That exact matching is stricter than fuzzy edit-distance matching and avoids accidental acceptance of short incorrect sequences. Levenshtein distance measures the number of edits needed to change one sequence into another, so allowing even one edit in a length-3 password can be overly permissive. [3]

## Gesture Labels

The current class mapping is:

```python
GESTURES = ["wave", "peace sign", "fist"]
```

The configured password is:

```python
PASSWORD_ORDER = [0, 1, 2]
# wave -> peace sign -> fist
```

## Requirements

- Python 3.10+ recommended.
- Linux, macOS, or Windows.
- Webcam connected and accessible.
- A trained `gesture_classifier.h5` model file that is only trained on "wave", "peace sign", "fist" gestures.
- An ngrok account and auth token if you want to expose the Flask API publicly. ngrok supports exposing local Flask development servers through a public tunnel, including via its Python SDK. [4]

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Required Files

Before running the system, make sure these files exist in the project root:

- `gesture_gui_http.py`
- `model_api.py`
- `gesture_classifier.h5`

If `gesture_classifier.h5` is missing, the API will start but inference will fail.

## Configure ngrok

Open `model_api.py` and replace the token with your own:

```python
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
```

Important security note:
- Do not commit your real ngrok token to GitHub. Attention!!!!!!
- If a token was ever exposed publicly, rotate it in your ngrok dashboard before using the repository again.

## Run the Project

### Step 1. Start the Flask API

Run this first:

```bash
python model_api.py
```

Expected output:

```text
✅ Model loaded
NGROK URL: https://<random-subdomain>.ngrok-free.dev
 * Running on http://127.0.0.1:5000
```

Copy the ngrok URL that appears in the terminal.

### Step 2. Update the GUI API URL

Open `gesture_gui_http.py` and set:

```python
API_URL = "https://<your-ngrok-url>.ngrok-free.dev"
```

The endpoints are built automatically:

```python
SEQUENCE_ENDPOINT = f"{API_URL}/sequence"
RESULT_ENDPOINT = f"{API_URL}/latest_result"
```

### Step 3. Start the GUI

In a second terminal with the same virtual environment activated:

```bash
python gesture_gui_http.py
```

### Step 4. Test the password

1. Click **Capture 2-second gesture**.
2. Perform the first gesture.
3. Repeat until three gestures are captured.
4. Watch the GUI labels:
   - `Current sequence: [...]`
   - `API result: ...`
   - `STATUS: WAITING / ACCESS GRANTED / ACCESS DENIED`

## API Endpoints

### `POST /sequence`
Accepts the current sequence and the captured landmark tensor.

Example request:

```bash
curl -X POST https://<your-ngrok-url>.ngrok-free.dev/sequence \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [0, 1],
    "landmarks": [[0.1, 0.2, -0.1], [0.0, 0.1, -0.2]]
  }'
```

Actual `landmarks` must be a full `(30, 63)` sequence.

Example response:

```json
{
  "timestamp": "2026-04-04T17:24:32.504700",
  "predicted_id": 2,
  "predicted_label": "fist",
  "decision": "WAITING",
  "current_sequence": [0, 1]
}
```

### `GET /latest_result`
Returns the most recent prediction result saved by the API.

Example:

```bash
curl https://<your-ngrok-url>.ngrok-free.dev/latest_result
```

## Typical Workflow After Cloning

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python model_api.py
# copy ngrok URL
# edit gesture_gui_http.py and paste API_URL
python gesture_gui_http.py
```

## Recommended GitHub Files

### `requirements.txt`
The included requirements file contains the Python packages needed to run the GUI and API.



## Troubleshooting

### GUI opens but no webcam image
- Check whether another app is already using the webcam.
- Confirm `cv2.VideoCapture(0)` points to the correct camera index.

### `Model not loaded`
- Verify `gesture_classifier.h5` is in the project root.
- Confirm the TensorFlow/Keras version can load the model.

### `Missing 'landmarks' in request`
- Make sure the GUI sends both `sequence` and `landmarks`.

### Access never gets granted
- Confirm the predicted classes truly match the configured password order.
- Print `self.current_sequence` in the GUI to verify sequence buildup.
- Reset the sequence before retrying a new password attempt.

### ngrok URL changed
- Every new tunnel may generate a new public URL unless you use a reserved domain.
- Update `API_URL` in `gesture_gui_http.py` whenever the tunnel URL changes. ngrok tunnels commonly expose a local app through a temporary public address. [4][7]

