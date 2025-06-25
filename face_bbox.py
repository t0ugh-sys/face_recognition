import os
from deepface import DeepFace
import cv2

# 配置路径（按需修改）
input_folder = r"D:\datasets\face\images"  # 输入图片文件夹
output_label_folder = r"D:\datasets\face\labels"  # YOLO标签输出文件夹
os.makedirs(output_label_folder, exist_ok=True)  # 创建输出目录[6](@ref)

# 选择高性能检测器（推荐以下两种）
#detector_backend = "retinaface"  # 高精度：适合复杂场景[4,5](@ref)
detector_backend = "yolov8"    # 速度优先：适合实时处理[2,4](@ref)

# 遍历所有图片文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', 'bmp')):
        img_path = os.path.join(input_folder, filename)

        # DeepFace人脸检测
        try:
            detections = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                enforce_detection=False  # 允许无人脸时不报错[5](@ref)
            )
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue

        # 读取图片尺寸用于归一化
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width, _ = img.shape

        # 准备YOLO标签内容
        yolo_lines = []
        for obj in detections:
            if "facial_area" not in obj:
                continue
            x = obj["facial_area"]["x"]
            y = obj["facial_area"]["y"]
            w = obj["facial_area"]["w"]
            h = obj["facial_area"]["h"]

            # 转换为YOLO格式（归一化中心点坐标和宽高）[6](@ref)
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height

            # 写入一行标签（class_id=0代表人脸）
            yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width_norm:.6f} {height_norm:.6f}")

        # 保存标签文件（与图片同名，扩展名为.txt）
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(output_label_folder, label_filename)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        print(f"已处理 {filename} -> 检测到 {len(yolo_lines)} 张人脸")