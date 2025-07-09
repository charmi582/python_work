import os

# 設定你的 YOLO 標註資料夾路徑
labels_folder = r'C:\Users\charmi\Desktop\2_label'

# 讀取 classes.txt
classes_file = os.path.join(labels_folder, 'classes.txt')
with open(classes_file, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

# 掃描資料夾內所有 .txt
for filename in os.listdir(labels_folder):
    if filename.endswith('.txt') and filename != 'classes.txt':
        file_path = os.path.join(labels_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) > 0:
                class_index = int(parts[0])
                if class_index < 0 or class_index >= len(classes):
                    print(f"❌ 檔案: {filename}, 行: {i+1}, index: {class_index} 不存在於 classes.txt 定義中")
