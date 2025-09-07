import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture('') #camera here

# Initialize MediaPipe Hands and Drawing utilities
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture video frame.")
        break

    img = cv2.flip(img, 1)

    # Convert to RGB
    try:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print(f"Error during color conversion: {e}")
        break

    results = hands.process(imgRGB)
    gesture = "Unknown"
    fingers = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get landmark coordinates
            landmarks = handLms.landmark
            h, w, c = img.shape

            # Detect raised fingers
            fingers = [
                # Thumb: Compare x-coordinates
                int(landmarks[mpHands.HandLandmark.THUMB_TIP].x < landmarks[mpHands.HandLandmark.THUMB_IP].x),
                # Other fingers: Compare y-coordinates
                int(landmarks[mpHands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mpHands.HandLandmark.INDEX_FINGER_DIP].y),
                int(landmarks[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mpHands.HandLandmark.MIDDLE_FINGER_DIP].y),
                int(landmarks[mpHands.HandLandmark.RING_FINGER_TIP].y < landmarks[mpHands.HandLandmark.RING_FINGER_DIP].y),
                int(landmarks[mpHands.HandLandmark.PINKY_TIP].y < landmarks[mpHands.HandLandmark.PINKY_DIP].y),
            ]

            # Detect specific gestures
            if sum(fingers) == 0:  # No fingers raised
                gesture = "None"
            elif fingers[1] == 1 and sum(fingers[2:]) == 0:  # pinky finger 
                gesture = "I"
            elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:  # ring finger
                gesture = "I LOVE"

            # Draw landmarks for fingertips
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id in [4, 8, 12, 16, 20]:  # Fingertip IDs
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Display gesture and finger count
            cv2.putText(img, f"Fingers: {sum(fingers)} | Gesture: {gesture}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        cv2.putText(img, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Detection", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
