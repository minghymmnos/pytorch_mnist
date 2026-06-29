# PyTorch MNIST 手写数字识别系统

基于 PyTorch 的 MNIST 手写数字识别项目，提供完整的 GUI 图形界面，支持模型训练、手写识别、模型管理三大功能。

## 项目结构

```
pytorch_mnist/
├── pytorch_mnist.py    # 核心模块：模型定义、训练、保存/加载、参数输出
├── mnist_gui.py        # 完整 GUI 应用（训练+识别+模型管理）
├── model.pth           # 训练好的模型权重（运行后生成）
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

程序自动下载 MNIST 数据集、训练模型、保存 `model.pth`、输出模型参数。

### 方式二：GUI 图形界面（推荐）

```bash
python mnist_gui.py
```

启动后有三个功能标签页：

## GUI 功能说明

### 1. 训练标签页

- **参数设置**：可调整迭代轮数(Epochs)、批次大小(Batch)、学习率(Learning Rate)
- **开始训练**：启动训练过程，实时显示进度条和训练日志
- **停止训练**：随时中断训练
- **重置模型**：将模型恢复为未训练的初始状态

### 2. 识别标签页

- 在 280×280 黑色画布上用鼠标手写数字
- 点击 **识别** 按钮，显示预测结果 + 0-9 置信度柱状条
- 点击 **清除** 按钮重置画布

### 3. 模型标签页

- **模型状态**：显示当前模型加载状态和路径
- **保存模型**：将当前模型保存到指定位置
- **加载模型**：从文件加载已有模型
- **加载默认模型**：加载 `model.pth`
- **查看参数**：显示模型各层的名称、形状、参数数量和数值

## 模型结构

| 层 | 输入 | 输出 | 激活函数 |
|------|-------|-------|----------------|
| fc1  | 784   | 64    | ReLU |
| fc2  | 64    | 64    | ReLU |
| fc3  | 64    | 64    | ReLU |
| fc4  | 64    | 10    | LogSoftmax |

总参数量: ~55,370

## API 说明

```python
from pytorch_mnist import Net, save_model, load_model, print_model_params, train_model

net = Net()
train_model(net, train_data, test_data, epochs=3, lr=0.001)  # 训练
save_model(net, "model.pth")      # 保存
load_model(net, "model.pth")      # 加载
print_model_params(net)           # 查看参数
```
