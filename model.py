import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 设置数据路径
train_path = "data/train"
test_path = "data/test"

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# 检查样本数据
image, label = train_dataset[0]
print(image)  # 图像张量
print(image.shape)
print(label)  # 图像标签


def imshow(img, title=None):
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean  # 反标准化
    img = torch.clamp(img, 0, 1)  # 将数据限制在[0,1]范围内
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.axis('off')


# 自定义规则类
class Rule1(nn.Module):
    def forward(self, x):
        x = torch.where(x < 0, torch.tensor(0.0, device=x.device), x)
        x = torch.where(x > 1, torch.tensor(1.0, device=x.device), x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bo1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bo2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self,x):
        out = f.relu(self.bo1(self.conv1(x)))
        out = self.bo2(self.conv2(out))
        out += self.shortcut(x)
        out = f.relu(out)
        return out


# 定义简单CNN模型1
class Simple1CNN(nn.Module):
    def __init__(self):
        super(Simple1CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu1 = Rule1()

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = f.relu(self.fc1(x))
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# 定义简单CNN模型2
class Simple2CNN(nn.Module):
    def __init__(self):
        super(Simple2CNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
        self.res_block1 = Residual(3, 16, 1)

    def forward(self, x):
        # x = self.pool(f.relu(self.conv1(x)))
        x= self.pool(f.relu(self.res_block1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义组合CNN模型
class CombineCNN(nn.Module):
    def __init__(self, model1cnn, model2cnn, output):
        super(CombineCNN, self).__init__()
        self.model1cnn = model1cnn
        self.model2cnn = model2cnn
        self.output_layer = nn.Sequential(
            nn.Linear(4, 128),  # 修改为 4 以适应 Simple1CNN 和 Simple2CNN 的输出拼接
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output),
        )

    def forward(self, x):
        x1 = self.model1cnn(x)
        x2 = self.model2cnn(x)
        x = torch.cat((x1, x2), 1)  # 拼接模型1和模型2的输出
        x = self.output_layer(x)
        return x

# 使用单一模型或组合模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Simple1CNN().to(device)
model = CombineCNN(Simple1CNN(), Simple2CNN(), 2).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    run_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
    print(f"Epoch: {epoch + 1}, Loss: {run_loss}")

# 测试模型
model.eval()
correct = 0
total = 0
predictions = []
true_labels = []
sample_images = []  # 保存样本图像
sample_labels = []  # 保存样本真实标签
sample_preds = []   # 保存样本预测标签

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        # 保存一个批次的样本数据用于展示
        if len(sample_images) < 8:  # 只保存8张图像
            sample_images.extend(inputs.cpu())
            sample_labels.extend(labels.cpu())
            sample_preds.extend(predicted.cpu())

print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

# 绘制预测效果图
fig = plt.figure(figsize=(12, 8))
for idx in range(8):  # 显示8张图片
    ax = fig.add_subplot(2, 4, idx+1)
    imshow(sample_images[idx])  # 显示图像
    ax.set_title(f"Pred: {sample_preds[idx].item()}, True: {sample_labels[idx].item()}")  # 显示预测和真实标签

plt.show()
