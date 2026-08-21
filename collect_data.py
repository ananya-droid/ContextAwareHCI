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


# =========================
# MEDIAPIPE SETUP
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
# LANDMARK EXTRACTION
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
# COLLECT ONE SAMPLE
# =========================

def collect_sample(label, sample_number):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    sequence = []

    print()
    print("--------------------------------")
    print("Preparing:", label)
    print("Sample:", sample_number)
    print("--------------------------------")
    print("Get your hand ready...")
    
    # Give the user 2 seconds
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

        cv2.imshow("Sign Language Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

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

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 × (x,y,z)
            landmarks = extract_landmarks(
                hand_landmarks
            )

            sequence.append(landmarks)

        # Display progress
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
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Sign Language Data Collector",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    # Convert to NumPy array
    sequence = np.array(sequence)

    # Create label folder
    label_dir = os.path.join(
        DATASET_DIR,
        label
    )

    os.makedirs(
        label_dir,
        exist_ok=True
    )

    # Save sample
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


# =========================
# MAIN PROGRAM
# =========================

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


print()
print("====================================")
print("SIGN LANGUAGE DATA COLLECTOR")
print("====================================")

print()
print("Available signs:")

for i, label in enumerate(labels, start=1):
    print(f"{i}. {label}")

print()
choice = input("Enter sign number: ")

try:
    choice = int(choice)

    if choice < 1 or choice > len(labels):
        print("Invalid choice.")
        exit()

except ValueError:
    print("Please enter a number.")
    exit()


label = labels[choice - 1]

samples = int(
    input("How many samples? ")
)

for sample_number in range(1, samples + 1):

    collect_sample(
        label,
        sample_number
    )

print()
print("====================================")
print("DATA COLLECTION COMPLETE")
print("====================================")