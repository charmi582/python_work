from ultralytics import YOLO
import serial
import time

# 建立串口連線（請根據你的 Jetson Nano port 修改）
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # 等待 ESP8266 初始化完成

# 載入你的模型
model = YOLO("D:/work/yolo11_transfer/best.pt")

# 使用 webcam 即時分類
for result in model.predict(source=0, stream=True, conf=0.5, show=True):
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0])  # 分類 ID
        class_name = names[class_id]  # 分類名稱（如 "recyclable"）

        print(f"偵測到：{class_name}")
        msg = class_name + "\n"
        ser.write(msg.encode())  # 傳送分類名稱給 ESP8266
