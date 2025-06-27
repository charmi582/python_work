import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # 防止 NumPy 報錯

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3

# 初始化語音引擎
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # 語速調整

# 初始化 MediaPipe 姿勢偵測
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 記數變數
shoot_count = 0
score = 0
shooting = False
hold_counter = 0
prev_wrist_y = None
speak_flag = {"shoot": False, "score": False}

# 攝影機畫面尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 虛擬籃框區域（畫面中央上方）
hoop_x1 = frame_width // 2 - 75
hoop_y1 = 50
hoop_x2 = frame_width // 2 + 75
hoop_y2 = 150

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 畫面處理
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # === 出手偵測 ===
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
            dy > 0.03
        ):
            hold_counter += 1
        else:
            hold_counter = 0
            shooting = False
            speak_flag["shoot"] = False

        if hold_counter > 3 and not shooting:
            shoot_count += 1
            shooting = True
            speak_flag["shoot"] = True

    # === 使用 Canny 邊緣檢測來識別藍球 ===
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:  # 篩選過小的輪廓
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # 顯示偵測到的物體

            # 檢查是否命中虛擬籃框
            if hoop_x1 < x < hoop_x2 and hoop_y1 < y < hoop_y2:
                score += 1
                speak_flag["score"] = True

    # === 播語音提示 ===
    if speak_flag["shoot"]:
        engine.say("Nice shot!")
        engine.runAndWait()
        speak_flag["shoot"] = False

    if speak_flag["score"]:
        engine.say("Scored!")
        engine.runAndWait()
        speak_flag["score"] = False

    # === 顯示統計資料 ===
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

    cv2.imshow("AImate - Voice Edition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

