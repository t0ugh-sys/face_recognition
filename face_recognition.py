# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from deepface import DeepFace
from ultralytics import YOLO
import logging
import torch
from PIL import Image
from torchvision import transforms

# ===== 配置参数 =====
DB_PATH = "face_db/"  # 人脸数据库路径
MIN_CONFIDENCE = 0.3  # YOLO检测置信度阈值
RECOG_THRESH = 0.35  # 人脸识别余弦相似度阈值
MIN_FACE_SIZE = 40  # 最小人脸尺寸(像素)
TARGET_FACE_SIZE = (224, 224)  # DeepFace输入尺寸
RESOLUTION = (640, 480)  # 摄像头分辨率
EMOTION_THRESHOLD = 0.0  # 情绪置信度阈值
YOLO_MODEL_PATH = "yolov8.pt"  # YOLOv8人脸检测模型[3,4](@ref)
EMOTION_MODEL_PATH = "minixception.pth"  # MiniXception模型路径
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Normal", "Sad", "Surprise"]  # 情绪标签

# 配置日志
logging.basicConfig(
    filename='recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# 创建数据库目录
os.makedirs(DB_PATH, exist_ok=True)

# 选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚙️ 使用设备: {device}")
logger.info(f"Using device: {device}")

# ===== 模型加载 =====
# 1. YOLOv8人脸检测模型[1,4](@ref)
try:
    model_yolo = YOLO(YOLO_MODEL_PATH)
    if device == 'cuda':
        model_yolo = model_yolo.cuda()
    print("✅ YOLOv8人脸检测模型加载成功")
    logger.info("YOLOv8 face detection model loaded")
except Exception as e:
    print(f"❌ YOLOv8加载失败: {e}")
    logger.error(f"YOLOv8 load error: {e}")
    exit()

# 2. DeepFace模型（轻量级SFace）
try:
    DeepFace.build_model("SFace")
    print("✅ DeepFace(SFace)模型加载成功")
    logger.info("DeepFace SFace model built")
except Exception as e:
    print(f"❌ DeepFace初始化失败: {e}")
    logger.error(f"DeepFace init error: {e}")
    exit()


# 3. MiniXception情绪识别模型
class MiniXception(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # 精简结构
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.pool1 = torch.nn.MaxPool2d(2)

        # 残差模块
        self.blocks = torch.nn.ModuleList()
        self.residual_convs = torch.nn.ModuleList()
        filters = [16, 32, 64, 128]
        in_ch = 8
        for f in filters:
            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, f, 3, padding=1, groups=in_ch),
                torch.nn.Conv2d(f, f, 1),
                torch.nn.BatchNorm2d(f),
                torch.nn.ReLU(),
                torch.nn.Conv2d(f, f, 3, padding=1, groups=f),
                torch.nn.Conv2d(f, f, 1),
                torch.nn.BatchNorm2d(f),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
            res_conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, f, 1, stride=2),
                torch.nn.BatchNorm2d(f)
            )
            self.blocks.append(block)
            self.residual_convs.append(res_conv)
            in_ch = f

        # 分类头
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(128, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        for block, res_conv in zip(self.blocks, self.residual_convs):
            residual = res_conv(x)
            x = block(x)
            if residual.shape[2:] != x.shape[2:]:
                residual = torch.nn.functional.interpolate(residual, size=x.shape[2:], mode='nearest')
            x = x + residual
            x = self.relu(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


try:
    model_emotion = MiniXception().to(device)
    state_dict = torch.load(EMOTION_MODEL_PATH, map_location=device)
    model_emotion.load_state_dict(state_dict)
    model_emotion.eval()
    print("✅ MiniXception情绪模型加载成功")
    logger.info("MiniXception model loaded")
except Exception as e:
    print(f"❌ MiniXception加载失败: {e}")
    logger.error(f"MiniXception load error: {e}")
    exit()

# ===== 向量数据库 =====
database = {}  # 格式: {人名: [嵌入向量列表]}


def init_vector_db():
    """从文件加载预存嵌入向量"""
    global database
    if os.path.exists(f"{DB_PATH}/embeddings.npy"):
        database = np.load(f"{DB_PATH}/embeddings.npy", allow_pickle=True).item()
        print(f"✅ 加载{len(database)}个人的嵌入向量")
        logger.info(f"Loaded {len(database)} embeddings")
    else:
        print("ℹ️ 未找到预存数据库，将从零创建")
        logger.info("No existing DB found")


def cosine_similarity(vec_a, vec_b):
    """计算余弦相似度（NumPy实现）"""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


# ===== 核心函数 =====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def get_face_embedding(face_img: np.ndarray) -> np.ndarray:
    """提取人脸嵌入向量（使用DeepFace）"""
    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, TARGET_FACE_SIZE)
        embedding = DeepFace.represent(
            img_path=face_rgb,
            model_name="SFace",
            detector_backend="skip",
            enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        return np.zeros(128, dtype=np.float32)  # 返回空向量


def recognize_face(face_img: np.ndarray):
    """基于余弦相似度识别人脸"""
    try:
        query_emb = get_face_embedding(face_img)
        best_match = ("Unknown", 0.0)

        # 遍历数据库所有人员
        for name, emb_list in database.items():
            for stored_emb in emb_list:
                similarity = cosine_similarity(query_emb, stored_emb)
                if similarity > best_match[1]:
                    best_match = (name, similarity)

        # 动态阈值调整（大脸更严格）
        face_area = face_img.shape[0] * face_img.shape[1]
        dynamic_thresh = max(0.25, RECOG_THRESH - 0.05 * (face_area / 10000))

        return best_match if best_match[1] > dynamic_thresh else ("Unknown", 0.0)
    except Exception as e:
        logger.error(f"识别异常: {str(e)}")
        return "Error", 0.0


def detect_emotion(face_img: np.ndarray):
    """使用MiniXception检测情绪"""
    try:
        if face_img.size == 0 or face_img.shape[0] < 10:
            return "Unknown", 0.0

        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model_emotion(face_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        emotion = EMOTION_LABELS[idx.item()]
        confidence = conf.item()
        return emotion, confidence
    except Exception as e:
        logger.error(f"情绪检测异常: {str(e)}")
        return "Error", 0.0


def enroll_face(face_img: np.ndarray, name: str):
    """注册人脸到数据库"""
    try:
        embedding = get_face_embedding(face_img)

        # 添加到数据库
        if name not in database:
            database[name] = []
        database[name].append(embedding)

        # 持久化存储
        np.save(f"{DB_PATH}/embeddings.npy", database)
        print(f"✅ {name} 注册成功 (总样本: {len(database[name])})")
        logger.info(f"Enrolled: {name} (samples: {len(database[name])})")
    except Exception as e:
        print(f"❌ 注册失败: {e}")
        logger.error(f"Enrollment failed: {e}")


# ===== 主程序 =====
def main():
    init_vector_db()  # 初始化数据库
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头初始化失败")
        logger.error("Camera init failed")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    fps_counter = []
    prev_time = time.time()

    print("\n【操作说明】")
    print("- 按 'Q': 退出程序")
    print("- 按 'E': 注册当前选中人脸（需输入姓名）")
    print("- 绿色框: 已知身份 | 红色框: 未知身份 | 青色文字: 情绪状态")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("帧读取失败")
            continue

        # 计算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        fps_counter.append(fps)
        avg_fps = np.mean(fps_counter[-10:]) if fps_counter else 0
        prev_time = curr_time

        # YOLO人脸检测（每帧处理）
        results = model_yolo(frame, conf=MIN_CONFIDENCE, verbose=False, imgsz=640)

        # 获取检测框并过滤无效结果
        boxes = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            # 过滤无效检测框
            valid_boxes = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                w, h = x2 - x1, y2 - y1

                # 尺寸/比例校验
                if (0.7 < w / h < 1.4 and  # 宽高比合理
                        w > MIN_FACE_SIZE and  # 最小宽度
                        h > MIN_FACE_SIZE and  # 最小高度
                        confs[i] > MIN_CONFIDENCE):  # 置信度阈值
                    valid_boxes.append((box, confs[i]))

            boxes = [b[0] for b in valid_boxes]

        # 处理每个检测到的人脸
        display_frame = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # 边界检查
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

            # 提取人脸区域
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:  # 跳过空图像
                continue

            # 同步处理识别和情绪检测
            identity, conf = recognize_face(face_img)
            emotion, emo_conf = detect_emotion(face_img)

            # 绘制结果
            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # 身份标签
            id_text = f"{identity[:12]}:{conf:.0%}" if identity != "Unknown" else "Unknown"
            cv2.putText(display_frame, id_text, (x1, y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 情绪标签
            if emo_conf > EMOTION_THRESHOLD:
                emo_text = f"{emotion[:8]}:{emo_conf:.0%}"
                cv2.putText(display_frame, emo_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # 显示帧率
        cv2.putText(display_frame, f"FPS: {int(avg_fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Multi-Face Recognition System", display_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e') and boxes:
            # 选择最大人脸注册
            largest_idx = np.argmax([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
            x1, y1, x2, y2 = map(int, boxes[largest_idx][:4])
            face_img = frame[y1:y2, x1:x2]
            name = input("请输入注册姓名: ").strip()
            if name:
                enroll_face(face_img, name)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 系统正常退出")
    logger.info("System shutdown")


if __name__ == "__main__":
    main()