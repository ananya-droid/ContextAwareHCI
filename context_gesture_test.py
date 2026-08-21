import cv2
import mediapipe as mp

from gesture_recognizer import recognize_gesture
from context_engine import get_action


# -------------------------
# CURRENT CONTEXT
# -------------------------

context = "PRESENTATION"


# -------------------------
# MEDIAPIPE SETUP
# -------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# -------------------------
# WEBCAM
# -------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Context-aware gesture test started.")
print("Current context:", context)
print("Press Q to quit.")


# -------------------------
# MAIN LOOP
# -------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    gesture = "NO HAND"
    action = "NO ACTION"

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Recognize gesture
        gesture = recognize_gesture(
            hand_landmarks.landmark
        )

        # Get context-specific action
        action = get_action(
            gesture,
            context
        )

    # -------------------------
    # DISPLAY
    # -------------------------

    cv2.putText(
        frame,
        f"Context: {context}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Action: {action}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "ContextAwareHCI",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------
# CLEANUP
# -------------------------

cap.release()
hands.close()
cv2.destroyAllWindows()