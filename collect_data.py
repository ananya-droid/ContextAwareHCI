import cv2
import numpy as np
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30
DATASET_DIR = "dataset"
MODEL_PATH = "hand_landmarker.task"


# ============================================================
# AVAILABLE SIGNS
# ============================================================

labels = [
    "HELLO",
    "YES",
    "NO",
    "PLEASE",
    "THANK_YOU",
    "HELP",
    "I",
    "YOU",
    "NEED",
    "WATER",
    "FOOD",
    "STOP",
    "GOOD",
    "BAD",
    "BYE"
]


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(hand_landmarks):

    landmarks = []

    for landmark in hand_landmarks:
        landmarks.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return landmarks


# ============================================================
# FIND NEXT SAMPLE NUMBER
# ============================================================

def get_next_sample_number(label):

    label_dir = os.path.join(DATASET_DIR, label)

    if not os.path.exists(label_dir):
        return 1

    existing_files = [
        f for f in os.listdir(label_dir)
        if f.startswith("sample_") and f.endswith(".npy")
    ]

    if not existing_files:
        return 1

    numbers = []

    for filename in existing_files:

        try:
            number = int(
                filename.replace("sample_", "")
                .replace(".npy", "")
            )

            numbers.append(number)

        except ValueError:
            pass

    if not numbers:
        return 1

    return max(numbers) + 1


# ============================================================
# COLLECT ONE SAMPLE
# ============================================================

def collect_sample(label, sample_number):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")

        return False

    sequence = []

    print()
    print("--------------------------------")
    print("Preparing:", label)
    print("Sample:", sample_number)
    print("--------------------------------")
    print("Get your hand ready...")

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
            "Sign Language Data Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            cap.release()
            cv2.destroyAllWindows()

            return False

    print("RECORDING!")

    while len(sequence) < SEQUENCE_LENGTH:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]

            # Extract 21 × (x,y,z)
            landmarks = extract_landmarks(
                hand_landmarks
            )

            if len(landmarks) == 63:

                sequence.append(landmarks)

            # Draw landmarks
            for landmark in hand_landmarks:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

        # Display sign
        cv2.putText(
            frame,
            f"Sign: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Display progress
        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to stop",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Sign Language Data Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            cap.release()
            cv2.destroyAllWindows()

            return False

    # ========================================================
    # SAVE SAMPLE
    # ========================================================

    sequence = np.array(sequence)

    # Safety check
    if sequence.shape != (30, 63):

        print(
            "ERROR: Incorrect sample shape:",
            sequence.shape
        )

        cap.release()
        cv2.destroyAllWindows()

        return False

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

    print()
    print("Saved:", filename)
    print("Shape:", sequence.shape)

    cap.release()
    cv2.destroyAllWindows()

    return True


# ============================================================
# MAIN PROGRAM
# ============================================================

print()
print("====================================")
print("SIGN LANGUAGE DATA COLLECTOR")
print("====================================")

print()
print("Available signs:")

for i, label in enumerate(labels, start=1):

    label_dir = os.path.join(
        DATASET_DIR,
        label
    )

    if os.path.exists(label_dir):

        count = len([
            f for f in os.listdir(label_dir)
            if f.endswith(".npy")
        ])

    else:

        count = 0

    print(
        f"{i:2d}. {label:<12} "
        f"({count} samples currently)"
    )


# ============================================================
# ASK WHICH SIGN
# ============================================================

print()

choice = input(
    "Enter the sign number to collect: "
)

try:

    choice = int(choice)

    if choice < 1 or choice > len(labels):

        print("Invalid choice.")

        exit()

except ValueError:

    print("Please enter a number.")

    exit()


label = labels[choice - 1]


# ============================================================
# SHOW CURRENT COUNT
# ============================================================

next_sample = get_next_sample_number(label)

current_count = next_sample - 1

print()
print("--------------------------------")
print("Selected sign:", label)
print("Current samples:", current_count)
print("Next sample:", next_sample)
print("--------------------------------")


# ============================================================
# ASK HOW MANY NEW SAMPLES
# ============================================================

samples = input(
    "How many NEW samples do you want to collect? "
)

try:

    samples = int(samples)

    if samples <= 0:

        print("Number of samples must be greater than 0.")

        exit()

except ValueError:

    print("Please enter a valid number.")

    exit()


# ============================================================
# COLLECT DATA
# ============================================================

print()
print(
    f"Collecting {samples} new samples "
    f"for {label}..."
)

print()

for i in range(samples):

    sample_number = next_sample + i

    success = collect_sample(
        label,
        sample_number
    )

    if not success:

        print()
        print("Data collection stopped.")

        break

    # Small pause between samples
    time.sleep(0.5)


# ============================================================
# COMPLETE
# ============================================================

print()
print("====================================")
print("DATA COLLECTION COMPLETE")
print("====================================")

print()
print("Sign:", label)

final_next = get_next_sample_number(label)

print(
    "Total samples now:",
    final_next - 1
)

print()
print("Run this to verify:")
print("python check_dataset.py")