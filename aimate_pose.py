import cv2
import mediapipe as mp

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 攝影機初始化
cap = cv2.VideoCapture(0)
shoot_count = 0
shooting = False
hold_counter = 0
prev_wrist_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換顏色格式
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 取得右手關鍵點
        landmarks = results.pose_landmarks.landmark
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        curr_wrist_y = r_wrist.y

        # 計算手腕Y軸上升速度
        if prev_wrist_y is not None:
            dy = prev_wrist_y - curr_wrist_y
        else:
            dy = 0
        prev_wrist_y = curr_wrist_y

        # 出手判定條件：
        # 1. 手腕與手肘高於肩膀
        # 2. 手腕高於手肘
        # 3. 手腕往上快速移動
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

        # 連續動作成立才判為一次出手
        if hold_counter > 3 and not shooting:
            shoot_count += 1
            shooting = True

    # 顯示出手次數
    cv2.putText(frame, f'Shoot count: {shoot_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AImate Pose Tracker v2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

