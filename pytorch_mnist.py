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


def get_data_loader(is_train, batch_size=15):
    """获取MNIST数据加载器"""
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def evaluate(test_data, net):
    """评估模型在测试集上的准确率"""
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
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


def train_model(net, train_data, test_data, epochs=2, lr=0.001, callback=None):
    """
    训练模型（可被 GUI 调用）
    参数:
        net: 模型
        train_data: 训练数据加载器
        test_data: 测试数据加载器
        epochs: 训练轮数
        lr: 学习率
        callback: 回调函数 callback(epoch, cur_epochs, accuracy, loss)
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    logs = []
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        acc = evaluate(test_data, net)
        msg = f"Epoch {epoch + 1}/{epochs}  损失: {avg_loss:.4f}  准确率: {acc:.2%}"
        logs.append(msg)
        if callback:
            callback(epoch + 1, epochs, acc, avg_loss)
    return logs


def main():
    """主函数: 训练模型、保存、加载、评估"""
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("=" * 40)
    print("训练前准确率: {:.2%}".format(evaluate(test_data, net)))

    logs = train_model(net, train_data, test_data, epochs=2)
    for log in logs:
        print(log)

    save_model(net, "model.pth")

    print(print_model_params(net))

    net2 = Net()
    load_model(net2, "model.pth")
    print("加载后模型准确率: {:.2%}".format(evaluate(test_data, net2)))

    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28), cmap="gray")
        plt.title("预测结果: " + str(int(predict)))
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
