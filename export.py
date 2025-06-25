# -*- coding:utf-8 -*-
# @Author  : t0ugh
# @Date    : 2025/6/18 11:58
# @Description:
# @Version : v1
import torch
import torch.nn as nn
class MiniXception(nn.Module):
    def __init__(self, num_classes=7):
        super(MiniXception, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 深度可分离卷积模块和对应的残差卷积
        self.blocks = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        filters = [16, 32, 64, 128]
        in_channels = 8
        for f in filters:
            block = nn.Sequential(
                nn.Conv2d(in_channels, f, kernel_size=3, padding=1, groups=in_channels),  # 深度卷积
                nn.Conv2d(f, f, kernel_size=1),  # 点卷积
                nn.BatchNorm2d(f),
                nn.ReLU(),
                nn.Conv2d(f, f, kernel_size=3, padding=1, groups=f),  # 深度卷积
                nn.Conv2d(f, f, kernel_size=1),  # 点卷积
                nn.BatchNorm2d(f),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            # 残差连接的卷积，确保通道数和空间尺寸匹配
            residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, f, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(f)
            )
            self.blocks.append(block)
            self.residual_convs.append(residual_conv)
            in_channels = f

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        for block, res_conv in zip(self.blocks, self.residual_convs):
            residual = res_conv(x)
            x = block(x)
            # 确保残差连接的空间尺寸匹配
            if residual.shape[2:] != x.shape[2:]:
                # 调整残差路径的空间尺寸
                residual = nn.functional.interpolate(residual, size=x.shape[2:], mode='nearest')
            x = x + residual

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 加载模型（补充权重加载校验）
model = MiniXception(num_classes=7)
try:
    state_dict = torch.load("minixception.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print("✅ 权重加载成功")
except Exception as e:
    print(f"❌ 权重加载失败: {str(e)}")
    exit(1)

# 重要：强制切换推理模式
model.eval()  # 关闭Dropout/BatchNorm的随机性[2,6](@ref)

# 创建示例输入
dummy_input = torch.randn(1, 1, 96, 96)  # 注意：输入是单通道96x96

# 导出模型
torch.onnx.export(
    model,
    dummy_input,
    "minixception.onnx",
    opset_version=12,  # 使用ONNX opset版本12
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)