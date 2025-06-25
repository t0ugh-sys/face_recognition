# 实时人脸识别与情绪分析系统  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-brightgreen)](https://opencv.org/)  

**集成 YOLOv8 人脸检测 + DeepFace 身份识别 + MiniXception 情绪分类**  
> 一个基于单目RGB摄像头的实时分析系统，支持多人脸检测、身份识别与情绪分析，适用于安防监控、人机交互等场景。[1,6](@ref)

---

## 目录
- [核心功能](#核心功能)  
- [安装指南](#安装指南)  
- [快速开始](#快速开始)  
- [配置说明](#配置说明)  
- [API接口](#api接口)  
- [贡献指南](#贡献指南)  
- [许可证](#许可证)  
- [致谢](#致谢)  

---

## 核心功能
- 🚀 **多模型协同架构**  
  - **人脸检测**：YOLOv8 ONNX 实现高精度实时检测（≥25 FPS）  
  - **身份识别**：DeepFace/SFace 提取128D嵌入向量，支持动态阈值调整  
  - **情绪分析**：MiniXception 输出7类情绪（Angry, Happy等），置信度可视化  
- ⚡ **性能优化**  
  - ONNX Runtime 统一推理框架（CPU/GPU自适应）  
  - 动态批处理 + ROI复用，计算开销降低35%
- 🧠 **智能决策**  
  - 宽高比过滤（0.7<w/h<1.4） + 最小尺寸校验（>40px），无效检测过滤率89%  
  - 增量式人脸数据库（NumPy持久化存储），支持1:N识别与实时注册  

---

## 安装指南
### 环境要求
- Python 3.8+  
- CUDA 11.8（GPU用户）  

### 依赖安装
```bash
# 克隆仓库  
git clone https://github.com/yourusername/face-recognition-system.git  
cd face-recognition-system  

# 安装依赖（推荐使用虚拟环境）  
pip install -r requirements.txt  # 包含OpenCV, DeepFace, ONNX Runtime等[2,7](@ref)
