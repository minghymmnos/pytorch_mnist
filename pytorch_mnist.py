# -*- coding: utf-8 -*-
"""
PyTorch MNIST 手写数字识别
- 训练模型并保存/加载
- 输出模型参数
- 评估模型准确率
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# ====== 模型文件配置 ======
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")       # 模型存放目录
MODEL_FILES = {"FNN": "fnn_model.pth", "CNN": "cnn_model.pth"}    # 不同模型的默认文件名


def ensure_model_dir():
    """确保 model 目录存在"""
    os.makedirs(MODEL_DIR, exist_ok=True)


def get_default_model_path(model_type="FNN"):
    """获取指定模型类型的默认保存路径"""
    filename = MODEL_FILES.get(model_type, MODEL_FILES["FNN"])
    return os.path.join(MODEL_DIR, filename)

# ====== 解决 matplotlib 中文显示问题 ======
_CN_FONTS_TRY = [
    "Microsoft YaHei",      # Windows
    "SimHei",               # Windows
    "WenQuanYi Micro Hei",  # Linux
    "Noto Sans CJK SC",     # Linux
    "Arial Unicode MS",     # macOS
]
for _f in _CN_FONTS_TRY:
    try:
        plt.rcParams["font.sans-serif"] = [_f]
        # 验证字体是否可用
        fig_test = plt.figure()
        fig_test.text(0, 0, "测试")
        plt.close(fig_test)
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
# ==========================================


class Net(torch.nn.Module):
    """4层全连接神经网络用于MNIST分类"""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


class CNNNet(torch.nn.Module):
    """CNN卷积神经网络用于MNIST分类
    结构: Conv1 → Conv2 → MaxPool → Dropout → FC1 → Dropout → FC2
    """

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)          # 输入1通道，输出32通道，3x3卷积核
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)          # 输入32通道，输出64通道，3x3卷积核
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)               # 64*12*12 = 9216
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        x = self.conv1(x)                                   # → (batch, 32, 26, 26)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)                                   # → (batch, 64, 24, 24)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)            # → (batch, 64, 12, 12)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)                             # → (batch, 9216)
        x = self.fc1(x)                                     # → (batch, 128)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)                                     # → (batch, 10)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output


# 模型类型注册表
MODEL_TYPES = {
    "FNN": Net,
    "CNN": CNNNet,
}


def create_model(model_type="FNN"):
    """
    根据类型创建模型实例
    参数:
        model_type: "FNN" 或 "CNN"
    返回:
        模型实例
    """
    cls = MODEL_TYPES.get(model_type)
    if cls is None:
        raise ValueError(f"不支持的模型类型: {model_type}，可选: {list(MODEL_TYPES.keys())}")
    return cls()


def detect_model_type(state_dict):
    """
    从state_dict中检测模型类型
    参数:
        state_dict: 模型状态字典
    返回:
        "CNN" 或 "FNN"
    """
    for key in state_dict.keys():
        if key.startswith("conv"):
            return "CNN"
    return "FNN"


def get_data_loader(is_train, batch_size=15):
    """获取MNIST数据加载器"""
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def evaluate(test_data, net, model_type="FNN"):
    """
    评估模型在测试集上的准确率
    参数:
        test_data: 测试数据加载器
        net: 模型
        model_type: "FNN" 或 "CNN"
    """
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            if model_type == "CNN":
                outputs = net.forward(x)          # CNN: 输入已经是 (batch, 1, 28, 28)
            else:
                outputs = net.forward(x.view(-1, 28 * 28))  # FNN: 需要展平
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def save_model(net, filepath="model.pth"):
    """
    保存模型权重到文件
    参数:
        net: 训练好的模型
        filepath: 保存路径（默认 model.pth）
    """
    torch.save(net.state_dict(), filepath)
    print(f"[✓] 模型已保存到: {filepath}")


def load_model(net, filepath="model.pth"):
    """
    从文件加载模型权重
    参数:
        net: 模型实例
        filepath: 模型文件路径
    返回:
        加载权重后的模型
    """
    if not os.path.exists(filepath):
        print(f"[!] 模型文件不存在: {filepath}")
        return net
    net.load_state_dict(torch.load(filepath, map_location="cpu"))
    print(f"[✓] 模型已从 {filepath} 加载")
    return net


def print_model_params(net):
    """
    输出模型所有层的参数信息
    包括: 层名称、参数形状、参数数值
    """
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("模型参数详情")
    lines.append("=" * 60)
    total_params = 0
    for name, param in net.named_parameters():
        num_params = param.numel()
        total_params += num_params
        lines.append(f"\n[层] {name}")
        lines.append(f"    形状: {list(param.shape)}")
        lines.append(f"    参数数量: {num_params:,}")
        flat = param.data.view(-1)
        if flat.numel() <= 10:
            lines.append(f"    数值: {flat.tolist()}")
        else:
            head = flat[:5].tolist()
            tail = flat[-5:].tolist()
            lines.append(f"    数值(前5): {head}")
            lines.append(f"    数值(后5): {tail}")
    lines.append("\n" + "-" * 60)
    lines.append(f"总参数数量: {total_params:,}")
    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


def train_model(net, train_data, test_data, epochs=2, lr=0.001, callback=None, model_type="FNN"):
    """
    训练模型（可被 GUI 调用）
    参数:
        net: 模型
        train_data: 训练数据加载器
        test_data: 测试数据加载器
        epochs: 训练轮数
        lr: 学习率
        callback: 回调函数 callback(epoch, cur_epochs, accuracy, loss)
        model_type: "FNN" 或 "CNN"
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    logs = []
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for (x, y) in train_data:
            net.zero_grad()
            if model_type == "CNN":
                output = net.forward(x)              # CNN: 输入 (batch, 1, 28, 28)
            else:
                output = net.forward(x.view(-1, 28 * 28))  # FNN: 展平
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        acc = evaluate(test_data, net, model_type=model_type)
        msg = f"Epoch {epoch + 1}/{epochs}  损失: {avg_loss:.4f}  准确率: {acc:.2%}"
        logs.append(msg)
        if callback:
            callback(epoch + 1, epochs, acc, avg_loss)
    return logs


def main():
    """主函数: 训练FNN和CNN模型、保存、评估"""
    ensure_model_dir()

    # ====== FNN 模型训练 ======
    print("\n" + "=" * 40)
    print("FNN 全连接神经网络训练")
    print("=" * 40)

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    fnn_net = create_model("FNN")

    print("训练前准确率: {:.2%}".format(evaluate(test_data, fnn_net, model_type="FNN")))

    logs = train_model(fnn_net, train_data, test_data, epochs=2, model_type="FNN")
    for log in logs:
        print(log)

    save_model(fnn_net, get_default_model_path("FNN"))
    print(print_model_params(fnn_net))

    # ====== CNN 模型训练 ======
    print("\n" + "=" * 40)
    print("CNN 卷积神经网络训练")
    print("=" * 40)

    cnn_net = create_model("CNN")

    print("训练前准确率: {:.2%}".format(evaluate(test_data, cnn_net, model_type="CNN")))

    logs = train_model(cnn_net, train_data, test_data, epochs=2, model_type="CNN")
    for log in logs:
        print(log)

    save_model(cnn_net, get_default_model_path("CNN"))
    print(print_model_params(cnn_net))

    # ====== 预测展示 ======
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        # FNN 预测
        predict_fnn = torch.argmax(fnn_net.forward(x[0].view(-1, 28 * 28)))
        # CNN 预测
        predict_cnn = torch.argmax(cnn_net.forward(x[0].view(1, 1, 28, 28)))

        plt.figure(n)
        plt.imshow(x[0].view(28, 28), cmap="gray")
        plt.title(f"FNN: {int(predict_fnn)}  CNN: {int(predict_cnn)}")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
