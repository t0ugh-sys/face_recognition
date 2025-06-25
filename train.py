import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载 数据集（替换为实际路径）
train_dataset = ImageFolder(root=r'D:\datasets\RAF-DB\RAF\train', transform=train_transform)
test_dataset = ImageFolder(root=r'D:\datasets\RAF-DB\RAF\val', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# MiniXception 模型定义
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


# 初始化模型、损失函数和优化器
model = MiniXception().to(device)
if os.path.exists('miniXception.pth'):
    model.load_state_dict(torch.load('miniXception.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%')

# 测试函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct/total:.2f}%')

# 训练和评估
train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'minixception.pth')

# 推理示例
def predict_emotion(image_path):
    model.eval()
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'normal', 'sad', 'surprise']
    image = torchvision.io.read_image(image_path).float()
    image = transforms.functional.resize(image, (48, 48))
    image = transforms.functional.to_grayscale(image)
    image = transforms.Normalize(mean=[0.5], std=[0.5])(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return emotions[pred]

# 示例推理
# print(predict_emotion('path_to_image.jpg'))