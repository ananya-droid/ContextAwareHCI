import cv2
import mediapipe as mp
import numpy as np
import os
import time


# =========================
# SETTINGS
# =========================

SEQUENCE_LENGTH = 30
DATASET_DIR = "dataset"

SAMPLES_PER_SIGN = 30


# =========================
# SIGNS
# =========================

labels = [
    "HELLO",
    "YES",
    "NO",
    "PLEASE",
    "THANK_YOU",
    "HELP",
    "NEED",
    "WATER",
    "FOOD",
    "BYE"
]


# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# =========================
# EXTRACT LANDMARKS
# =========================

def extract_landmarks(hand_landmarks):

    landmarks = []

    for landmark in hand_landmarks.landmark:

        landmarks.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return landmarks


# =========================
# RECORD ONE SAMPLE
# =========================

def collect_sample(label, sample_number):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return False

    sequence = []

    print()
    print("--------------------------------")
    print(f"Sign: {label}")
    print(f"Sample: {sample_number}/{SAMPLES_PER_SIGN}")
    print("--------------------------------")

    # Preparation countdown
    start_time = time.time()

    while time.time() - start_time < 2:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            f"Get ready: {label}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Sign Language Dataset Collector",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            cap.release()
            cv2.destroyAllWindows()

            return False

    print("RECORDING!")

    # Record 30 frames
    while len(sequence) < SEQUENCE_LENGTH:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = extract_landmarks(
                hand_landmarks
            )

            sequence.append(landmarks)

        cv2.putText(
            frame,
            f"Sign: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Sample: {sample_number}/{SAMPLES_PER_SIGN}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Sign Language Dataset Collector",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            cap.release()
            cv2.destroyAllWindows()

            return False

    # =========================
    # SAVE DATA
    # =========================

    sequence = np.array(sequence)

    label_dir = os.path.join(
        DATASET_DIR,
        label
    )

    os.makedirs(
        label_dir,
        exist_ok=True
    )

    filename = os.path.join(
        label_dir,
        f"sample_{sample_number:03d}.npy"
    )

    np.save(
        filename,
        sequence
    )

    print("Saved:", filename)

    cap.release()
    cv2.destroyAllWindows()

    return True


# =========================
# MAIN
# =========================

print()
print("======================================")
print("SIGN LANGUAGE DATASET COLLECTOR")
print("======================================")

print()
print("Signs to collect:")

for i, label in enumerate(labels, start=1):
    print(f"{i}. {label}")

print()
print("Each sign will collect:")
print(f"{SAMPLES_PER_SIGN} samples")
print(f"{SEQUENCE_LENGTH} frames per sample")

print()
print("Press Q during recording to stop.")


# =========================
# COLLECT ALL SIGNS
# =========================

for label in labels:

    print()
    print("======================================")
    print(f"NOW COLLECTING: {label}")
    print("======================================")

    label_dir = os.path.join(
        DATASET_DIR,
        label
    )

    os.makedirs(
        label_dir,
        exist_ok=True
    )

    # Find existing samples
    existing_files = [
        f for f in os.listdir(label_dir)
        if f.endswith(".npy")
    ]

    start_sample = len(existing_files) + 1

    for sample_number in range(
        start_sample,
        SAMPLES_PER_SIGN + 1
    ):

        success = collect_sample(
            label,
            sample_number
        )

        if not success:

            print("Collection stopped.")

            hands.close()

            exit()

print()
print("======================================")
print("ALL DATA COLLECTION COMPLETE")
print("======================================")

hands.close()