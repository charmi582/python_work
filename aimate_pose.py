import cv2
import mediapipe as mp

# 初始化 MediaPipe 模組
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 開啟攝影機
cap = cv2.VideoCapture(0)
shoot_count = 0
shooting = False  # 防止重複記錄

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換顏色格式
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 取得關鍵點
        landmarks = results.pose_landmarks.landmark
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        # 判斷是否舉手投籃：手腕高於肩膀 + 手腕快速上升
        if right_wrist_y < right_shoulder_y and not shooting:
            shoot_count += 1
            shooting = True
        elif right_wrist_y > right_elbow_y:
            shooting = False  # 動作結束，等待下一次投籃

    # 顯示出手次數
    cv2.putText(frame, f'Shoot count: {shoot_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AImate Shooting Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
