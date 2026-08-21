import cv2
import mediapipe as mp
import time

from action_controller import execute_action
from gesture_recognizer import recognize_gesture
def perform_action(gesture):
    if gesture == "THUMBS UP":
        return "CONFIRM"

    elif gesture == "FIST":
        return "CANCEL"

    elif gesture == "PEACE":
        return "NEXT"

    elif gesture == "POINT":
        return "SELECT"

    elif gesture == "OPEN PALM":
        return "PAUSE"

    else:
        return "NO ACTION"


# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Gesture test started.")
print("Show your hand to the camera.")
print("Press Q to quit.")



while True:

    success, frame = cap.read()
    

    if not success:
        print("Could not read camera frame.")
        break

    # Flip image so it behaves like a mirror
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    gesture = "NO HAND"

    if results.multi_hand_landmarks:

        # Get first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw the 21 landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Send landmarks to our gesture recognizer
        gesture = recognize_gesture(
            hand_landmarks.landmark
        )
        
    action=perform_action(gesture)
    if action != "NO ACTION":
       execute_action(action)
       time.sleep(1)
    # Display gesture
    cv2.putText(
        frame,
        f"Gesture: {gesture}|Action:{action}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Show camera
    cv2.imshow("ContextAwareHCI - Gesture Test", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()

print("Gesture test stopped.")
