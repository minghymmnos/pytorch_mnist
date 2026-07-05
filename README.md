# PyTorch MNIST 手写数字识别系统

基于 PyTorch 的 MNIST 手写数字识别项目，提供完整的 GUI 图形界面，支持**两种模型**（全连接网络 FNN / 卷积网络 CNN）的训练、手写识别、模型管理三大功能。

## 项目结构

```
pytorch_mnist/
├── pytorch_mnist.py    # 核心模块：模型定义(FNN/CNN)、训练、保存/加载、参数输出
├── mnist_gui.py        # 完整 GUI 应用（训练+识别+模型管理）
├── model/              # 模型权重文件存放目录
│   ├── fnn_model.pth   # 训练好的 FNN 模型权重
│   └── cnn_model.pth   # 训练好的 CNN 模型权重
├── README.md           # 项目文档
└── MNIST/              # MNIST 数据集（自动下载）
```

## 环境要求

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- Pillow (PIL)
- tkinter（Python 内置，Linux 需额外安装 `python3-tk`）

安装依赖：

```bash
pip install torch torchvision matplotlib pillow
```

Linux 用户还需安装 tkinter：

```bash
sudo apt-get install python3-tk
```

## 使用方式

### 方式一：命令行训练

```bash
python pytorch_mnist.py
```

程序自动下载 MNIST 数据集，依次训练 **FNN**（全连接网络）和 **CNN**（卷积网络）两个模型，分别保存到 `model/fnn_model.pth` 和 `model/cnn_model.pth`，输出模型参数并展示预测结果。

### 方式二：GUI 图形界面（推荐）

```bash
python mnist_gui.py
```

启动后有三个功能标签页：

## GUI 功能说明

### 1. 训练标签页

- **模型类型选择**：可在训练前选择 **FNN (全连接网络)** 或 **CNN (卷积网络)**
  - FNN：训练速度快，适合快速验证
  - CNN：准确率更高（可达 99%+），训练稍慢
- **参数设置**：可调整迭代轮数(Epochs)、批次大小(Batch)、学习率(Learning Rate)
- **开始训练**：启动训练过程，实时显示进度条和训练日志
- **停止训练**：随时中断训练
- **重置模型**：将模型恢复为未训练的初始状态

### 2. 识别标签页

- 在 280×280 黑色画布上用鼠标手写数字
- 点击 **识别** 按钮，显示预测结果 + 0-9 置信度柱状条
- 点击 **清除** 按钮重置画布
- 自动适应当前加载的模型类型（FNN/CNN）

### 3. 模型标签页

- **模型状态**：显示当前模型类型和加载状态
- **保存模型**：将当前模型保存到指定位置
- **加载模型**：从文件加载已有模型（自动检测 FNN/CNN 类型）
- **加载默认模型**：加载 `model.pth`
- **查看参数**：显示模型各层的名称、形状、参数数量和数值

## 模型结构

### FNN (全连接神经网络)

| 层 | 输入 | 输出 | 激活函数 |
|------|-------|-------|----------------|
| fc1  | 784   | 64    | ReLU |
| fc2  | 64    | 64    | ReLU |
| fc3  | 64    | 64    | ReLU |
| fc4  | 64    | 10    | LogSoftmax |

总参数量: ~55,370

### CNN (卷积神经网络)

| 层 | 输入尺寸 | 输出尺寸 | 参数 |
|------|------------|-------------|------|
| Conv1 (3×3) | 1×28×28 | 32×26×26 | 320 |
| ReLU | 32×26×26 | 32×26×26 | - |
| Conv2 (3×3) | 32×26×26 | 64×24×24 | 18,496 |
| ReLU | 64×24×24 | 64×24×24 | - |
| MaxPool (2×2) | 64×24×24 | 64×12×12 | - |
| Dropout (0.25) | 64×12×12 | 64×12×12 | - |
| Flatten | 64×12×12 | 9,216 | - |
| FC1 | 9,216 | 128 | 1,179,776 |
| ReLU | 128 | 128 | - |
| Dropout (0.5) | 128 | 128 | - |
| FC2 | 128 | 10 | 1,290 |
| LogSoftmax | 10 | 10 | - |

总参数量: ~1,199,882

> CNN 通过卷积层提取空间特征，参数量虽多但准确率显著优于 FNN。

## API 说明

```python
from pytorch_mnist import (
    Net, CNNNet, create_model, evaluate,
    save_model, load_model, print_model_params, train_model,
    detect_model_type, MODEL_TYPES
)

# 创建模型
fnn_net = create_model("FNN")     # 全连接网络
cnn_net = create_model("CNN")     # 卷积网络

# 训练
train_model(fnn_net, train_data, test_data, epochs=3, lr=0.001, model_type="FNN")
train_model(cnn_net, train_data, test_data, epochs=3, lr=0.001, model_type="CNN")

# 评估
acc = evaluate(test_data, cnn_net, model_type="CNN")

# 保存/加载
save_model(cnn_net, "cnn_model.pth")
load_model(cnn_net, "cnn_model.pth")

# 查看参数
print_model_params(cnn_net)

# 检测模型类型
state_dict = torch.load("model.pth", map_location="cpu")
model_type = detect_model_type(state_dict)  # 返回 "FNN" 或 "CNN"
```
