import cv2
import mediapipe as mp
import math
import random
particles = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            lm_list = []
            h, w, c = frame.shape

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Thumb tip (4) and Index tip (8)
            if len(lm_list) >= 9:
                x1, y1 = lm_list[4]  # thumb tip
                x2, y2 = lm_list[8]  # index finger
                x3, y3 = lm_list[12] # middle finger
                # Draw glowing triangle
                for i in range(1, 5):
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), i*2)
                    cv2.line(frame, (x2, y2), (x3, y3), (255, 0, 255), i*2)
                    cv2.line(frame, (x3, y3), (x1, y1), (255, 0, 255), i*2)

                # Bright core triangle
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 255), 2)
                cv2.line(frame, (x3, y3), (x1, y1), (255, 255, 255), 2)
                # Add particles near fingers
                for _ in range(5):
                    particles.append([x2, y2, random.randint(-5,5), random.randint(-5,5)])

                # Update particles
                for p in particles:
                     p[0] += p[2]
                     p[1] += p[3]
                     cv2.circle(frame, (p[0], p[1]), 3, (255, 0, 255), -1)

                     # Limit particles
                     if len(particles) > 100:
                         particles = particles[-100:]


                # Draw circles
                cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)

                # Draw line
                # Glow effect (multiple layers)
                for i in range(1, 6):
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), i*2)
                # Bright core line
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)    

    
                # Distance
                distance = math.hypot(x2 - x1, y2 - y1)

            # Pinch detection
            if distance < 40:
                cv2.putText(frame, "POWER", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 3)
                
                # Strong glow when pinching
                blur = cv2.GaussianBlur(frame, (25, 25), 0)
                frame = cv2.addWeighted(frame, 1, blur, 1, 0)
                
                
                

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Create glow effect using blur
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    frame = cv2.addWeighted(frame, 1, blur, 0.6, 0)
    cv2.imshow("Pinch Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()