# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from ultralytics import YOLO
import logging
import pickle

# ===== 配置参数 =====
DB_PATH = "face_db"  # 人脸数据库目录
MIN_CONFIDENCE = 0.3  # 人脸检测置信度阈值
RECOG_THRESH = 0.45  # 识别阈值（可动态调整）
MIN_FACE_SIZE = 40  # 最小有效人脸尺寸
TARGET_SIZE = (160, 160)  # 人脸对齐尺寸
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Normal", "Sad", "Surprise"]  # 情绪标签
REGISTER_THRESH = 0.7  # 注册时重复人脸阈值

# ===== 初始化设置 =====
os.makedirs(DB_PATH, exist_ok=True)
logging.basicConfig(filename='face_system.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


# ===== 人脸数据库类 =====
class FaceDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.encodings_file = os.path.join(db_path, "encodings.pkl")
        self.known_encodings = []
        self.known_names = []
        self._load_database()

    def _load_database(self):
        """加载已有的人脸数据"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data['encodings']
                    self.known_names = data['names']
                print(f"✅ 已加载{len(set(self.known_names))}人的人脸数据")
            except Exception as e:
                print(f"❌ 数据库加载失败: {e}")

    def save_database(self):
        """保存数据库到文件"""
        data = {'encodings': self.known_encodings, 'names': self.known_names}
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)

    def add_face(self, name: str, embedding: np.ndarray):
        """添加新人脸到数据库"""
        # 查重校验
        for stored_emb in self.known_encodings:
            if np.linalg.norm(embedding - stored_emb) < 0.4:  # L2距离阈值
                return False, "相似人脸已存在"

        self.known_encodings.append(embedding)
        self.known_names.append(name)
        self.save_database()
        return True, f"{name}注册成功"

    def recognize(self, query_embedding: np.ndarray):
        """识别人脸并返回最佳匹配"""
        best_match = ("Unknown", 0.0)
        if not self.known_encodings:
            return best_match

        # 计算余弦相似度
        for name, stored_emb in zip(self.known_names, self.known_encodings):
            norm_q = np.linalg.norm(query_embedding)
            norm_s = np.linalg.norm(stored_emb)
            similarity = np.dot(query_embedding, stored_emb) / (norm_q * norm_s)

            if similarity > best_match[1]:
                best_match = (name, similarity)

        return best_match


# ===== 核心功能函数 =====
def get_face_embedding(face_img: np.ndarray) -> np.ndarray:
    """提取人脸特征向量"""
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, TARGET_SIZE)
    embedding = DeepFace.represent(
        img_path=face_img,
        model_name="SFace",
        detector_backend="skip",
        enforce_detection=False
    )[0]["embedding"]
    return np.array(embedding)


def align_face(image, box, expand_ratio=0.15):
    """人脸对齐并扩展区域"""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # 扩展检测框
    x1 = max(0, x1 - int(w * expand_ratio))
    y1 = max(0, y1 - int(h * expand_ratio))
    x2 = min(image.shape[1], x2 + int(w * expand_ratio))
    y2 = min(image.shape[0], y2 + int(h * expand_ratio))

    return image[y1:y2, x1:x2]


def display_register_menu(frame, faces):
    """显示注册选择界面"""
    display_frame = frame.copy()
    for i, (box, face_img) in enumerate(faces):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(display_frame, f"{i + 1}", (x1 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(display_frame, "选择要注册的人脸编号 (1-9)", 10, 30,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, "按ESC取消注册", 10, 70,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return display_frame


# ===== 主程序 =====
def main():
    # 初始化组件
    face_db = FaceDatabase(DB_PATH)
    model_yolo = YOLO("yolov8.pt")  # 人脸检测模型
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 状态变量
    register_mode = False
    candidate_faces = []

    print("操作指南:")
    print("- 按 'R': 进入注册模式")
    print("- 按 'Q': 退出系统")
    print("- 注册模式下: 输入数字选择对应人脸")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # 人脸检测
        results = model_yolo(frame, conf=MIN_CONFIDENCE, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # 处理有效人脸
        valid_faces = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1

            # 过滤无效人脸
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE: continue
            if abs(w / h - 1) > 0.3: continue  # 宽高比过滤

            face_img = align_face(frame, box)
            valid_faces.append((box, face_img))

            # 非注册模式实时识别
            if not register_mode:
                embedding = get_face_embedding(face_img)
                name, similarity = face_db.recognize(embedding)

                # 动态调整阈值（大脸更严格）
                face_area = w * h
                dynamic_thresh = max(0.35, RECOG_THRESH - 0.05 * (face_area / 10000))
                status = name if similarity > dynamic_thresh else "Unknown"

                # 绘制结果
                color = (0, 255, 0) if status != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{status[:12]}:{similarity:.0%}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 注册模式处理
        if register_mode:
            if not valid_faces:
                print("⚠️ 未检测到有效人脸，注册取消")
                register_mode = False
            else:
                candidate_faces = valid_faces
                display_frame = display_register_menu(frame, candidate_faces)
                cv2.imshow("人脸注册", display_frame)

                # 获取键盘选择
                key = cv2.waitKey(20)
                if 49 <= key <= 57:  # 数字1-9
                    idx = key - 49
                    if idx < len(candidate_faces):
                        selected_box, selected_face = candidate_faces[idx]
                        name = input("请输入姓名: ").strip()
                        if name:
                            embedding = get_face_embedding(selected_face)
                            success, msg = face_db.add_face(name, embedding)
                            print(msg)
                        register_mode = False
                elif key == 27:  # ESC
                    register_mode = False

        # 显示实时画面
        if not register_mode:
            cv2.imshow("多人脸识别系统", frame)

        # 全局按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            register_mode = True
            candidate_faces = []
            print("进入注册模式...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()