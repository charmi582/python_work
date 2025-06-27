import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  

import cv2
import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


shoot_count = 0
score = 0
shooting = False
hold_counter = 0
prev_wrist_y = None


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


hoop_x1 = frame_width // 2 - 75
hoop_y1 = 50
hoop_x2 = frame_width // 2 + 75
hoop_y2 = 150

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

   
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        curr_wrist_y = r_wrist.y
        dy = prev_wrist_y - curr_wrist_y if prev_wrist_y else 0
        prev_wrist_y = curr_wrist_y

        if (
            r_wrist.y < r_shoulder.y and
            r_elbow.y < r_shoulder.y and
            r_wrist.y < r_elbow.y and
            dy > 0.02
        ):
            hold_counter += 1
        else:
            hold_counter = 0
            shooting = False

        if hold_counter > 3 and not shooting:
            shoot_count += 1
            shooting = True


    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 140, 255), 2)
            if hoop_x1 < x < hoop_x2 and hoop_y1 < y < hoop_y2:
                score += 1

    accuracy = (score / shoot_count) * 100 if shoot_count > 0 else 0.0

    cv2.rectangle(frame, (hoop_x1, hoop_y1), (hoop_x2, hoop_y2), (255, 0, 0), 2)
    cv2.putText(frame, "Virtual Hoop", (hoop_x1, hoop_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, f'Shoots: {shoot_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Score: {score}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Accuracy: {accuracy:.1f}%', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("AImate Tracker Combined", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
