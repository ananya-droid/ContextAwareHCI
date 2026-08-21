import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "sign_model.keras"
LABEL_PATH = "label_classes.npy"
HAND_MODEL = "hand_landmarker.task"

SEQUENCE_LENGTH = 30


# ==========================================
# LOAD LSTM MODEL
# ==========================================

print("Loading LSTM model...")

model = tf.keras.models.load_model(MODEL_PATH)

labels = np.load(
    LABEL_PATH,
    allow_pickle=True
)

print("Classes:", labels)


# ==========================================
# CREATE MEDIAPIPE HAND LANDMARKER
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    detector.close()
    exit()

print()
print("==============================")
print("LIVE SIGN RECOGNITION")
print("==============================")
print("Show HELLO or YES.")
print("HELLO = your recorded HELLO sign")
print("YES   = thumbs up")
print("Press Q to quit.")
print()


# ==========================================
# VARIABLES
# ==========================================

sequence = []

prediction = "NO HAND"
confidence = 0.0

timestamp_ms = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe expects an mp.Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp must increase
    timestamp_ms += 33

    # Detect hand
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ======================================
    # HAND FOUND
    # ======================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        landmarks = []

        # 21 landmarks
        for landmark in hand:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Add frame to sequence
        sequence.append(landmarks)

        # Keep only latest 30 frames
        if len(sequence) > SEQUENCE_LENGTH:

            sequence = sequence[-SEQUENCE_LENGTH:]


        # ==================================
        # DRAW LANDMARKS
        # ==================================

        h, w, _ = frame.shape

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        # ==================================
        # LSTM PREDICTION
        # ==================================

        if len(sequence) == SEQUENCE_LENGTH:

            input_data = np.array(
                sequence,
                dtype=np.float32
            )

            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            probabilities = model.predict(
                input_data,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            prediction = str(
                labels[predicted_index]
            )

            confidence = float(
                probabilities[predicted_index]
            )


    # ======================================
    # NO HAND
    # ======================================

    else:

        sequence = []

        prediction = "NO HAND"
        confidence = 0.0


    # ======================================
    # DISPLAY
    # ======================================

    cv2.rectangle(
        frame,
        (10, 10),
        (450, 150),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Sign: {prediction}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Context-Aware Sign Language Recognition",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
detector.close()
cv2.destroyAllWindows()

print("Recognition stopped.")