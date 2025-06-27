import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # 避免 NumPy 出錯（重要）

import cv2
import mediapipe as mp
import csv

# 初始化 MediaPipe 姿勢模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 建立資料夾與 CSV 檔案
os.makedirs("pose_data", exist_ok=True)
csv_file = open("pose_data/shooting_pose_data.csv", mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)

# 寫入標題：33 個點的 x, y，再加上 label 欄位
headers = []
for i in range(33):  # MediaPipe 共 33 個關鍵點
    headers += [f"x_{i}", f"y_{i}"]
headers.append("label")  # 標註用欄位
csv_writer.writerow(headers)

# 開啟攝影機
cap = cv2.VideoCapture(0)
label = None

print("🎯 資料蒐集開始！請做出出手動作後：")
print("按下 G → 標註為『好姿勢』")
print("按下 B → 標註為『壞姿勢』")
print("按下 Q → 結束並儲存檔案")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 擷取 33 個點的 x, y 值
        pose_row = []
        for lm in results.pose_landmarks.landmark:
            pose_row.extend([lm.x, lm.y])

        # 顯示提示
        cv2.putText(frame, "Press G (good) or B (bad) to label pose", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 如果有標記，寫入 CSV
        if label:
            pose_row.append(label)
            csv_writer.writerow(pose_row)
            print(f"✅ 已記錄一筆姿勢資料：{label}")
            label = None

    cv2.imshow("AImate Pose Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'):
        label = "good"
    elif key == ord('b'):
        label = "bad"
    elif key == ord('q'):
        break

# 關閉資源
cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("✅ 資料收集完成！已儲存至：pose_data/shooting_pose_data.csv")
