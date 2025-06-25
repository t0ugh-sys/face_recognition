import os
import cv2
import numpy as np

# ====== 配置区域 ======
IMAGE_DIR = "D:/datasets/face/images"  # 图片目录路径
LABEL_DIR = "D:/datasets/face/labels"  # 标签目录路径
CLASS_NAMES = ["face"]  # 类别名称列表
FONT_SCALE = 0.5  # 字体大小
THICKNESS = 1  # 文本粗细
COLOR = (0, 255, 0)  # 文本颜色 (BGR格式)


# =====================

def draw_yolo_annotations(cv_img, annotations):
    """
    使用OpenCV在图片上绘制YOLO标注
    """
    h, w = cv_img.shape[:2]

    # 为不同类别分配颜色
    class_colors = {}
    for ann in annotations:
        class_id = int(ann[0])
        if class_id not in class_colors:
            class_colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())

    # 绘制每个检测框
    for ann in annotations:
        class_id = int(ann[0])
        cx, cy, bw, bh = map(float, ann[1:5])

        # 转换为像素坐标
        x = int((cx - bw / 2) * w)
        y = int((cy - bh / 2) * h)
        width = int(bw * w)
        height = int(bh * h)

        # 绘制矩形框
        cv2.rectangle(cv_img, (x, y), (x + width, y + height), class_colors[class_id], 2)

        # 获取文本尺寸 (使用OpenCV原生方法)
        label = f"{class_id}:{CLASS_NAMES[class_id]}" if class_id < len(CLASS_NAMES) else str(class_id)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS
        )

        # 绘制文本背景
        cv2.rectangle(
            cv_img,
            (x, y - text_height - 5),
            (x + text_width, y),
            class_colors[class_id],
            -1
        )

        # 添加类别文本
        cv2.putText(
            cv_img, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), THICKNESS
        )

    return cv_img


def display_images_with_annotations():
    """ 主函数：批量显示带标注的图片 """
    # 检查目录存在性
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(LABEL_DIR):
        print(f"错误：目录不存在！请检查路径配置")
        return

    # 获取所有图片文件
    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print("未找到图片文件！")
        return

    print(f"找到 {len(image_files)} 张图片，按任意键继续，ESC退出...")

    # 遍历并显示每张图片
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_file)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_file)[0] + ".txt")

        # 读取图片 (直接使用OpenCV)
        cv_img = cv2.imread(img_path)
        if cv_img is None:
            print(f"无法读取图片: {img_file}")
            continue

        # 检查标签文件
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                annotations = [line.strip().split() for line in f.readlines() if line.strip()]

        # 绘制标注
        if annotations:
            cv_img = draw_yolo_annotations(cv_img, annotations)

        # 显示图片信息
        status = f"{i + 1}/{len(image_files)} | detect target: {len(annotations)}"
        cv2.putText(cv_img, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示图片
        cv2.imshow("YOLO Annotation Viewer", cv_img)

        # 键盘控制
        key = cv2.waitKey(0)
        if key == 27:  # ESC退出
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    display_images_with_annotations()