# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from deepface import DeepFace
import onnxruntime as ort
import logging
from PIL import Image
from torchvision import transforms
from scipy.special import softmax

# ===== 配置参数 =====
DB_PATH = "face_db/"  # 人脸数据库路径
MIN_CONFIDENCE = 0.3  # YOLO检测置信度阈值
RECOG_THRESH = 0.5  # 人脸识别余弦相似度阈值
MIN_FACE_SIZE = 40  # 最小人脸尺寸(像素)
TARGET_FACE_SIZE = (224, 224)  # DeepFace输入尺寸
RESOLUTION = (640, 480)  # 摄像头分辨率
EMOTION_THRESHOLD = 0.0  # 情绪置信度阈值（保持为0以强制显示）
YOLO_MODEL_PATH = "yolov8.onnx"  # YOLOv8 ONNX人脸检测模型
EMOTION_MODEL_PATH = "minixception.onnx"  # MiniXception ONNX模型路径
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
device = 'cuda' if ort.get_device() == 'GPU' else 'cpu'
print(f"⚙️ 使用设备: {device}")
logger.info(f"Using device: {device}")

# ===== 模型加载 =====
# 1. YOLOv8 ONNX人脸检测模型
try:
    yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
    print("✅ YOLOv8 ONNX人脸检测模型加载成功")
    logger.info("YOLOv8 ONNX face detection model loaded")
except Exception as e:
    print(f"❌ YOLOv8 ONNX加载失败: {e}")
    logger.error(f"YOLOv8 ONNX load error: {e}")
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

# 3. MiniXception ONNX情绪识别模型
try:
    emotion_session = ort.InferenceSession(EMOTION_MODEL_PATH, providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
    input_name = emotion_session.get_inputs()[0].name
    input_shape = emotion_session.get_inputs()[0].shape
    output_name = emotion_session.get_outputs()[0].name
    print(f"✅ MiniXception ONNX情绪模型加载成功, 输入: {input_name} 形状: {input_shape}, 输出: {output_name}")
    logger.info(f"MiniXception ONNX model loaded, input: {input_name}, shape: {input_shape}, output: {output_name}")
except Exception as e:
    print(f"❌ MiniXception ONNX加载失败: {e}")
    logger.error(f"MiniXception ONNX load error: {e}")
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
    transforms.Resize((96 ,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_yolo_input(frame, img_size=640):
    """预处理YOLOv8 ONNX输入"""
    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_yolo_output(outputs, img_shape, conf_thres=0.3, iou_thres=0.45):
    """处理YOLOv8 ONNX输出（假设输出为[1, 5, 8400]），添加NMS"""
    boxes = []
    scores = []
    output = outputs[0]  # [1, 5, 8400]
    output = output.transpose((0, 2, 1))  # [1, 8400, 5]
    output = output[0]  # [8400, 5]
    for det in output:
        conf = det[4]
        if conf < conf_thres:
            continue
        x_center, y_center, w, h = det[:4]
        x1 = int((x_center - w / 2) * img_shape[1] / 640)
        y1 = int((y_center - h / 2) * img_shape[0] / 640)
        x2 = int((x_center + w / 2) * img_shape[1] / 640)
        y2 = int((y_center + h / 2) * img_shape[0] / 640)
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)

    boxes = np.array(boxes)
    scores = np.array(scores)

    if len(boxes) > 0:
        # 应用NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
        if len(indices) > 0:
            return [boxes[i] for i in indices]
    return boxes

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
        return np.array(embedding, dtype=np.float32) # 每个人脸对应128的向量
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
    """使用MiniXception ONNX检测情绪"""
    try:
        if face_img.size == 0 or face_img.shape[0] < 10:
            return "Unknown", 0.0

        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).numpy()
        face_tensor = np.expand_dims(face_tensor, axis=0)  # [1, 1, 96, 96]
        input_name = emotion_session.get_inputs()[0].name
        inputs = {input_name: face_tensor}

        outputs = emotion_session.run(None, inputs)
        if not outputs or len(outputs) == 0:
            raise ValueError("ONNX模型输出为空")
        probs = softmax(outputs[0], axis=1)[0]  # 假设第一个输出是logits
        conf = np.max(probs)
        idx = np.argmax(probs)

        emotion = EMOTION_LABELS[idx] if 0 <= idx < len(EMOTION_LABELS) else "Unknown"
        confidence = conf
        logger.info(f"Emotion detection: {emotion}, confidence: {confidence}, probs: {probs}")
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

        # YOLOv8 ONNX人脸检测
        input_tensor = preprocess_yolo_input(frame)
        outputs = yolo_session.run(None, {yolo_session.get_inputs()[0].name: input_tensor})
        boxes = postprocess_yolo_output(outputs, frame.shape, MIN_CONFIDENCE, iou_thres=0.45)

        # 过滤无效检测框
        valid_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if (0.7 < w / h < 1.4 and  # 宽高比合理
                    w > MIN_FACE_SIZE and  # 最小宽度
                    h > MIN_FACE_SIZE):  # 最小高度
                valid_boxes.append(box)

        # 处理每个检测到的人脸
        display_frame = frame.copy()
        for box in valid_boxes:
            x1, y1, x2, y2 = map(int, box)

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

            # 身份标签（显示在上方）
            id_text = f"{identity[:12]}:{conf:.0%}" if identity != "Unknown" else "Unknown"
            cv2.putText(display_frame, id_text, (x1, y1 - 25 if y1 - 25 > 0 else y1 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 情绪标签（显示在下方）
            if emo_conf >= 0:  # 强制显示情绪
                emo_text = f"{emotion[:8]}:{emo_conf:.0%}"
                cv2.putText(display_frame, emo_text, (x1, y2 + 20 if y2 + 20 < frame.shape[0] else y2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示帧率
        cv2.putText(display_frame, f"FPS: {int(avg_fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Multi-Face Recognition System", display_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e') and valid_boxes:
            # 选择最大人脸注册
            largest_idx = np.argmax([(box[2] - box[0]) * (box[3] - box[1]) for box in valid_boxes])
            x1, y1, x2, y2 = map(int, valid_boxes[largest_idx])
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
    try:
        main()
    except KeyboardInterrupt as e:
        print("🔌 系统正常退出")
        logger.info("System shutdown")