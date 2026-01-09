"""
案例:
    演示CNN的综合案例, 图像分类.

回顾: 深度学习项目的步骤
    1. 准备数据集.
        这里我们用的时候 计算机视觉模块 torchvision自带的 CIFAR10数据集, 包含6W张 (32,32,3)的图片, 5W张训练集, 1W张测试集, 10个分类, 每个分类6K张图片.
        你需要单独安装一下 torchvision包, 即: pip install torchvision
    2. 搭建(卷积)神经网络
    3. 模型训练.
    4. 模型测试.

卷积层:
    提取图像的局部特征 -> 特征图(Feature Map), 计算方式:  N = (W - F + 2P) // S + 1
    每个卷积核都是1个神经元.

池化层:
    降维, 有最大池化 和 平均池化.
    池化只在HW上做调整, 通道上不改变.

案例的优化思路:
    1. 增加卷积核的输出通道数(大白话: 卷积核的数量)
    2. 增加全连接层的参数量.
    3. 调整学习率
    4. 调整优化方法(optimizer...)
    5. 调整激活函数...
    6. ...
"""

# 导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8


# 1. 准备数据集.
def create_dataset():
    # 1. 获取训练集.
    # 参1: 数据集路径. 参2: 是否是训练集. 参3: 数据预处理 -> 张量数据. 参4: 是否联网下载(直接用我给的, 不用下)
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    # 2. 获取测试集.
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    # 3. 返回数据集.
    return train_dataset, test_dataset


# 2. 搭建(卷积)神经网络
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
      
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 0)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 0)
        self.bn2 = nn.BatchNorm2d(32)
       
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)

       
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        
        self.linear1 = nn.Linear(2304, 120) 
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        # 第1组: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)      # 归一化要在激活之前
        x = torch.relu(x)
        x = self.pool1(x)

        # 第2组: Conv -> BN -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        # 第3组: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(-1, 2304)
        
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)  # Dropout 用在全连接层
        x = torch.relu(self.linear2(x))
        x = self.output(x)
        
        return x


# 3. 模型训练.
def train(train_dataset):
    # 1. 创建数据加载器.
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 2. 创建模型对象.
    model = ImageModel()
    # 3. 创建损失函数对象.
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数 = softmax()激活函数 + 损失计算.
    # 4. 创建优化器对象.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 5. 循环遍历epoch, 开始 每轮的 训练动作.
    # 5.1 定义变量, 记录训练的总轮数.
    epochs = 20
    # 5.2 遍历, 完成每轮的 所有批次的 训练动作.
    for epoch_idx in range(epochs):
        # 5.2.1 定义变量, 记录: 总损失, 总样本数据量, 预测正确样本个数, 训练(开始)时间
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        # 5.2.2 遍历数据加载器, 获取到 每批次的 数据.
        for x, y in dataloader:
            # 5.2.3 切换训练模式.
            model.train()
            # 5.2.4 模型预测.
            y_pred = model(x)
            # 5.2.5 计算损失.
            loss = criterion(y_pred, y)
            # 5.2.6 梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5.2.7 统计预测正确的样本个数.
            # print(y_pred)     # 批次中, 每张图 每个分类的 预测概率.

            # argmax() 返回最大值对应的索引, 充当 -> 该图片的 预测分类.
            # tensor([9, 8, 5, 5, 1, 5, 8, 5])
            # print(torch.argmax(y_pred, dim=-1))     # -1这里表示行.   预测分类
            # print(y)                                # 真实分类
            # print(torch.argmax(y_pred, dim=-1) == y)    # 是否预测正确
            # print((torch.argmax(y_pred, dim=-1) == y).sum())    # 预测正确的样本个数.
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()

            # 5.2.8 统计当前批次的总损失.          第1批平均损失 * 第1批样本个数
            total_loss += loss.item() * len(y)  # [第1批总损失 +  第2批总损失 +  第3批总损失 +  ...]
            # 5.2.9  统计当前批次的总样本个数.
            total_samples += len(y)
            # break   每轮只训练1批, 提高训练效率, 减少训练时长, 只有测试会这么写, 实际开发绝不要这样做.

        # 5.2.10 走这里, 说明一轮训练完毕, 打印该轮的训练信息.
        print(f'epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.2f}, time:{time.time() - start:.2f}s')
        # break     # 这里写break, 意味着只训练一轮.

    # 6. 保存模型.
    torch.save(model.state_dict(), './model/image_model.pth')

# 4. 模型测试.
def evaluate(test_dataset):
    # 1. 创建测试集 数据加载器.
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 2. 创建模型对象.
    model = ImageModel()
    # 3. 加载模型参数.
    model.load_state_dict(torch.load('./model/image_model.pth'))    # pickle文件
    # 4. 定义变量统计 预测正确的样本个数, 总样本个数.
    total_correct, total_samples = 0, 0
    # 5. 遍历数据加载器, 获取到 每批次 的数据.
    for x, y in dataloader:
        # 5.1 切换模型模式.
        model.eval()
        # 5.2 模型预测.
        y_pred = model(x)
        # 5.3 因为训练的时候用了CrossEntropyLoss, 所以搭建神经网络时没有加softmax()激活函数, 这里要用 argmax()来模拟.
        # argmax()函数功能:  返回最大值对应的索引, 充当 -> 该图片的 预测分类.
        y_pred = torch.argmax(y_pred, dim=-1)   # -1 这里表示行.
        # 5.4 统计预测正确的样本个数.
        total_correct += (y_pred == y).sum()
        # 5.5 统计总样本个数.
        total_samples += len(y)

    # 6. 打印正确率(预测结果).
    print(f'Acc: {total_correct / total_samples:.2f}')
def train1(train_dataset):
    model = ImageModel()
    epochs = 10
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    for epoch in range(epochs):
        total_loss=0
        total_samples=0
        total_correct= 0.0
        start = time.time()
        for x, y in dataloader:
            y_pred = model(x)# [16, 10]
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(y) 
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_samples += len(y)
        print(f"第{epoch+1}轮的总损失为：{total_loss / total_samples:.2f} 准确率： {total_correct / total_samples:.2f} 总时长为: {time.time() - start:.2f}")
        scheduler.step()
    torch.save(model.state_dict(), './model/image_model.pth')
    print('保存模型成功')
def evaluate1(test_dataset):    
    dataloader = DataLoader(test_dataset, batch_size=16)
    model = ImageModel()
    model.load_state_dict(torch.load('./model/image_model.pth'))
    total_correct = 0
    total_samples = 0
    for x, y in dataloader:
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=-1)
        total_correct += (y_pred == y).sum()
        total_samples += len(y)
    print(f'准确率：{total_correct / total_samples:.2f}')
# 5. 测试
if __name__ == '__main__':
    # 1. 获取数据集.
    train_dataset, test_dataset = create_dataset()
    # 2. 搭建神经网络.
    # model = ImageModel()
    # 查看模型参数, 参1: 模型, 参2: 输入维度(CHW, 通道, 高, 宽), 参3: 批次大小
    # summary(model, (3, 32, 32), batch_size=1)

    # 3. 模型训练.
    # train1(train_dataset)
    # evaluate(test_dataset)

    # 4. 模型测试.
    evaluate1(test_dataset)
