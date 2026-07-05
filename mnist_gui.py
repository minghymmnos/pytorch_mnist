# -*- coding: utf-8 -*-
"""
MNIST 手写数字识别 - 完整 GUI 应用
功能: 模型训练、手写识别、模型管理（保存/加载/查看参数）
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import numpy as np

from pytorch_mnist import (
    Net,
    CNNNet,
    create_model,
    MODEL_TYPES,
    detect_model_type,
    MODEL_DIR,
    MODEL_FILES,
    ensure_model_dir,
    get_default_model_path,
    get_data_loader,
    evaluate,
    save_model,
    load_model,
    print_model_params,
    train_model,
)


# ========================================================================
#  自定义文本输出控件 - 将 print 重定向到 GUI 文本框
# ========================================================================
class TextRedirector:
    """将 print 输出重定向到 tkinter Text 组件"""

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass


# ========================================================================
#  主应用
# ========================================================================
class MNISTApp:
    """MNIST 完整 GUI 应用"""

    CANVAS_WIDTH = 280
    CANVAS_HEIGHT = 280
    PEN_SIZE = 16

    # ==================== 初始化 ====================

    def __init__(self, master):
        self.master = master
        master.title("MNIST 手写数字识别系统")
        master.resizable(False, False)

        # ---------- 确保模型目录 ----------
        ensure_model_dir()

        # ---------- 模型 ----------
        self.model_type = "FNN"  # 默认模型类型
        self.net = create_model(self.model_type)
        self.model_path = get_default_model_path(self.model_type)
        self.model_loaded = self._try_load_model()

        # ---------- 训练状态 ----------
        self.training = False
        self.train_data = None
        self.test_data = None

        # ---------- 构建界面 ----------
        self._build_ui()
        self._update_model_status()

    def _try_load_model(self):
        """尝试加载已有模型"""
        if os.path.exists(self.model_path):
            try:
                load_model(self.net, self.model_path)
                return True
            except Exception:
                return False
        return False

    # ==================== 构建界面 ====================

    def _build_ui(self):
        # 使用 ttk.Notebook 实现分页
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_train_tab()
        self._build_recognize_tab()
        self._build_model_tab()

        # 底部状态栏
        self.status_label = tk.Label(
            self.master, text="就绪",
            bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

    # ==================== Tab 1: 训练 ====================

    def _build_train_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" 训练 ")

        # --- 参数设置 ---
        param_box = ttk.LabelFrame(frame, text="训练参数", padding=10)
        param_box.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(param_box)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="迭代轮数(Epochs):", width=14).pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=3)
        ttk.Spinbox(row1, from_=1, to=20, textvariable=self.epochs_var, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="  批次大小(Batch):", width=14).pack(side=tk.LEFT)
        self.batch_var = tk.IntVar(value=15)
        ttk.Spinbox(row1, from_=1, to=128, textvariable=self.batch_var, width=8).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(param_box)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="学习率(Learning Rate):", width=14).pack(side=tk.LEFT)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(row2, from_=0.0001, to=0.1, increment=0.0001,
                    textvariable=self.lr_var, width=8).pack(side=tk.LEFT, padx=5)

        # 模型类型选择
        row3 = ttk.Frame(param_box)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="模型类型:", width=14).pack(side=tk.LEFT)
        self.model_type_var = tk.StringVar(value="FNN")
        model_type_combo = ttk.Combobox(
            row3, textvariable=self.model_type_var,
            values=["FNN (全连接网络)", "CNN (卷积网络)"],
            state="readonly", width=20
        )
        model_type_combo.pack(side=tk.LEFT, padx=5)
        model_type_combo.bind("<<ComboboxSelected>>", self._on_model_type_changed)
        ttk.Label(row3, text="  CNN准确率更高，训练稍慢", foreground="gray").pack(side=tk.LEFT)

        # --- 操作按钮 ---
        btn_box = ttk.Frame(frame)
        btn_box.pack(fill=tk.X, pady=5)

        self.train_btn = ttk.Button(btn_box, text="开始训练", command=self._start_training)
        self.train_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = ttk.Button(btn_box, text="停止训练", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(btn_box, text="重置模型", command=self._reset_model)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # --- 进度条 ---
        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.pack(fill=tk.X, pady=5)
        self.progress_label = ttk.Label(frame, text="")
        self.progress_label.pack(anchor=tk.W)

        # --- 日志 ---
        ttk.Label(frame, text="训练日志:").pack(anchor=tk.W)
        log_frame = ttk.Frame(frame)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.train_log = tk.Text(log_frame, height=12, width=80, state=tk.NORMAL,
                                 font=("Consolas", 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.train_log.yview)
        self.train_log.configure(yscrollcommand=scrollbar.set)
        self.train_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _start_training(self):
        """启动训练（在新线程中执行）"""
        if self.training:
            return

        self.training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.train_log.delete(1.0, tk.END)
        self.progress["value"] = 0
        self.progress_label.config(text="准备中...")
        self.status_label.config(text="正在训练...")

        # 预加载数据
        self.train_log.insert(tk.END, "[*] 加载数据...\n")
        self.train_log.update()
        try:
            self.train_data = get_data_loader(is_train=True, batch_size=self.batch_var.get())
            self.test_data = get_data_loader(is_train=False, batch_size=self.batch_var.get())
        except Exception as e:
            messagebox.showerror("错误", f"数据加载失败: {e}")
            self.training = False
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return

        # 重置模型参数（使用新模型开始训练）
        selected = self.model_type_var.get()
        model_type = "CNN" if "CNN" in selected else "FNN"
        self.model_type = model_type
        self.train_net = create_model(model_type)
        epochs = self.epochs_var.get()
        lr = self.lr_var.get()

        # 在线程中执行训练
        def train_thread():
            # 训练前评估
            init_acc = evaluate(self.test_data, self.train_net, model_type=model_type)
            self._append_log(f"[*] 初始准确率: {init_acc:.2%}\n")
            self._append_log(f"[*] 开始训练: epochs={epochs}, batch={self.batch_var.get()}, lr={lr}, model={model_type}\n\n")

            logs = train_model(
                self.train_net, self.train_data, self.test_data,
                epochs=epochs, lr=lr,
                callback=self._on_epoch_end,
                model_type=model_type,
            )

            if self.training:  # 没有被停止
                self._append_log("\n[*] 训练完成!\n")
                self.master.after(0, self._on_train_complete)

        t = threading.Thread(target=train_thread, daemon=True)
        t.start()

    def _on_epoch_end(self, epoch, total, acc, loss):
        """每个 epoch 结束时的回调"""
        def update():
            if not self.training:
                return
            pct = (epoch / total) * 100
            self.progress["value"] = pct
            self.progress_label.config(text=f"Epoch {epoch}/{total}  准确率: {acc:.2%}  损失: {loss:.4f}")
        self.master.after(0, update)

    def _append_log(self, text):
        """在主线程中追加日志"""
        def update():
            self.train_log.insert(tk.END, text)
            self.train_log.see(tk.END)
        self.master.after(0, update)

    def _on_train_complete(self):
        """训练完成"""
        self.training = False
        self.train_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="训练完成")
        self.status_label.config(text="训练完成")

        # 将训练好的模型设为主模型
        self.net.load_state_dict(self.train_net.state_dict())
        self.model_path = get_default_model_path(self.model_type)
        self.model_loaded = True
        # 同步模型类型下拉框
        self.model_type_var.set("CNN (卷积网络)" if self.model_type == "CNN" else "FNN (全连接网络)")
        self._update_model_status()

    def _stop_training(self):
        """停止训练"""
        self.training = False
        self.train_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="训练已停止")
        self.status_label.config(text="训练已停止")
        self._append_log("\n[!] 训练已手动停止\n")

    def _on_model_type_changed(self, event=None):
        """模型类型切换时的处理"""
        selected = self.model_type_var.get()
        new_type = "CNN" if "CNN" in selected else "FNN"
        if new_type != self.model_type:
            if messagebox.askyesno("切换模型类型",
                                   f"切换将重置当前模型为新的{new_type}模型，是否继续？"):
                self.model_type = new_type
                self.net = create_model(self.model_type)
                self.model_path = get_default_model_path(self.model_type)
                self.model_loaded = False
                self._update_model_status()
                self.status_label.config(text=f"已切换至{new_type}模型")
            else:
                # 恢复选择
                old_label = "CNN (卷积网络)" if self.model_type == "CNN" else "FNN (全连接网络)"
                self.model_type_var.set(old_label)

    def _reset_model(self):
        """重置模型为未训练状态"""
        if messagebox.askyesno("确认", "确定要重置模型为未训练的初始状态吗？"):
            self.net = create_model(self.model_type)
            self.model_path = get_default_model_path(self.model_type)
            self.model_loaded = False
            self._update_model_status()
            self.status_label.config(text="模型已重置")

    # ==================== Tab 2: 识别 ====================

    def _build_recognize_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" 识别 ")

        # 左侧：画布
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            left_frame,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            bg="black",
            cursor="crosshair",
        )
        self.canvas.pack()

        # 同步 PIL 图像缓冲区
        self._reset_image_buffer()

        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.last_x = None
        self.last_y = None

        # 按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.clear_btn = ttk.Button(btn_frame, text="清除", command=self._clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.recognize_btn = ttk.Button(btn_frame, text="识别", command=self._recognize)
        self.recognize_btn.pack(side=tk.RIGHT)

        # 右侧：结果显示
        right_frame = ttk.Frame(frame, padding=(20, 0, 0, 0))
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(right_frame, text="识别结果", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        self.result_label = ttk.Label(
            right_frame, text="?", font=("Arial", 48, "bold"),
            anchor=tk.CENTER, width=3
        )
        self.result_label.pack(pady=(0, 20))

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(right_frame, text="置信度分布", font=("Arial", 12, "bold")).pack(pady=(0, 5))

        # 置信度条
        self.bars = []
        self.bar_labels = []
        for i in range(10):
            row = ttk.Frame(right_frame)
            row.pack(anchor=tk.W, pady=1)
            ttk.Label(row, text=f"{i}:", width=2, anchor=tk.E).pack(side=tk.LEFT)
            bar = tk.Canvas(row, width=200, height=16, bg="#eee", highlightthickness=0)
            bar.pack(side=tk.LEFT, padx=3)
            val_label = ttk.Label(row, text="0.0%", width=6, anchor=tk.W)
            val_label.pack(side=tk.LEFT)
            self.bars.append(bar)
            self.bar_labels.append(val_label)

    def _reset_image_buffer(self):
        """重置 PIL 图像缓冲区"""
        self.image = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), 0)
        self.draw = ImageDraw.Draw(self.image)

    def _on_mouse_down(self, event):
        self.last_x, self.last_y = event.x, event.y
        self._draw_point(event.x, event.y)

    def _on_mouse_move(self, event):
        if self.last_x is not None and self.last_y is not None:
            x, y = event.x, event.y
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill="white", width=self.PEN_SIZE,
                capstyle=tk.ROUND, smooth=True,
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255, width=self.PEN_SIZE,
            )
            self.last_x, self.last_y = x, y

    def _on_mouse_up(self, event):
        self.last_x = None
        self.last_y = None

    def _draw_point(self, x, y):
        r = self.PEN_SIZE // 2
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill="white", outline="white",
        )
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _clear_canvas(self):
        self.canvas.delete("all")
        self._reset_image_buffer()
        self.result_label.config(text="?")
        for bar in self.bars:
            bar.delete("all")
        for lbl in self.bar_labels:
            lbl.config(text="0.0%")
        self.status_label.config(text="画布已清除")

    def _recognize(self):
        if not self.model_loaded:
            messagebox.showerror("错误", "模型未加载！\n请先在「训练」选项卡中训练模型，\n或在「模型」选项卡中加载已有模型。")
            return

        img_small = self.image.resize((28, 28), Image.LANCZOS)
        arr = np.array(img_small, dtype=np.float32) / 255.0

        with torch.no_grad():
            self.net.eval()
            if self.model_type == "CNN":
                # CNN: 输入形状 (1, 1, 28, 28)
                tensor = torch.tensor(arr, dtype=torch.float32).view(1, 1, 28, 28)
            else:
                # FNN: 输入形状 (1, 784)
                tensor = torch.tensor(arr, dtype=torch.float32).view(1, 28 * 28)
            output = self.net.forward(tensor)
            probs = F.softmax(output, dim=1).squeeze(0).numpy()
            pred = int(torch.argmax(output, dim=1).item())

        self.result_label.config(text=str(pred))
        self.status_label.config(text=f"识别完成，预测结果: {pred}")

        for i in range(10):
            pct = probs[i] * 100
            self.bars[i].delete("all")
            self.bars[i].create_rectangle(0, 0, int(pct * 2), 16, fill="#4CAF50", outline="")
            self.bar_labels[i].config(text=f"{pct:.1f}%")

    # ==================== Tab 3: 模型 ====================

    def _build_model_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=" 模型 ")

        # --- 模型状态 ---
        self.model_status_var = tk.StringVar(value="未加载模型")
        status_box = ttk.LabelFrame(frame, text="模型状态", padding=10)
        status_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_box, textvariable=self.model_status_var,
                  font=("Arial", 10, "bold")).pack(anchor=tk.W)

        # --- 操作按钮 ---
        btn_box = ttk.LabelFrame(frame, text="操作", padding=10)
        btn_box.pack(fill=tk.X, pady=(0, 10))

        btn_row1 = ttk.Frame(btn_box)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="保存模型", command=self._save_model_ui).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row1, text="加载模型", command=self._load_model_ui).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row1, text="加载默认模型", command=self._load_default_model).pack(side=tk.LEFT, padx=5)

        # --- 查看参数 ---
        param_box = ttk.LabelFrame(frame, text="模型参数", padding=10)
        param_box.pack(fill=tk.BOTH, expand=True)

        ttk.Button(param_box, text="刷新参数信息", command=self._refresh_params).pack(anchor=tk.W, pady=(0, 5))

        self.params_text = tk.Text(param_box, height=18, width=80, state=tk.NORMAL,
                                   font=("Consolas", 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(param_box, orient=tk.VERTICAL, command=self.params_text.yview)
        self.params_text.configure(yscrollcommand=scrollbar.set)
        self.params_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _update_model_status(self):
        """更新模型状态显示"""
        model_label = {"FNN": "全连接网络(FNN)", "CNN": "卷积网络(CNN)"}.get(self.model_type, self.model_type)
        if self.model_loaded:
            self.model_status_var.set(f"✅ {model_label} 已加载 ({self.model_path})")
            self.status_label.config(text="模型就绪")
        else:
            self.model_status_var.set(f"❌ {model_label} 未加载（请训练或加载模型）")
            self.status_label.config(text="模型未加载")

    def _save_model_ui(self):
        """UI 保存模型"""
        default_name = MODEL_FILES.get(self.model_type, "fnn_model.pth")
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch 模型", "*.pth"), ("所有文件", "*.*")],
            initialdir=MODEL_DIR,
            initialfile=default_name,
            title="保存模型",
        )
        if filepath:
            save_model(self.net, filepath)
            self.model_path = filepath
            messagebox.showinfo("成功", f"模型已保存到:\n{filepath}")

    def _load_model_ui(self):
        """UI 加载模型"""
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch 模型", "*.pth"), ("所有文件", "*.*")],
            title="加载模型",
        )
        if filepath:
            try:
                # 检测模型类型
                state_dict = torch.load(filepath, map_location="cpu")
                model_type = detect_model_type(state_dict)
                new_net = create_model(model_type)
                new_net.load_state_dict(state_dict)
                self.net = new_net
                self.model_type = model_type
                self.model_path = filepath
                self.model_loaded = True
                # 同步下拉框
                self.model_type_var.set("CNN (卷积网络)" if model_type == "CNN" else "FNN (全连接网络)")
                self._update_model_status()
                messagebox.showinfo("成功", f"已加载 {model_type} 模型:\n{filepath}")
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {e}")

    def _load_default_model(self):
        """加载当前模型类型的默认模型（位于 model/ 目录）"""
        path = get_default_model_path(self.model_type)
        if not os.path.exists(path):
            # 尝试加载另一个类型的模型作为备选
            alt_type = "CNN" if self.model_type == "FNN" else "FNN"
            alt_path = get_default_model_path(alt_type)
            if os.path.exists(alt_path):
                path = alt_path
                # 临时更新类型
                self.model_type = alt_type
                self.model_type_var.set("CNN (卷积网络)" if alt_type == "CNN" else "FNN (全连接网络)")
            else:
                messagebox.showwarning("警告", f"默认模型文件不存在: {path}\n请先在「训练」选项卡中训练对应模型。")
                return
        try:
            state_dict = torch.load(path, map_location="cpu")
            model_type = detect_model_type(state_dict)
            new_net = create_model(model_type)
            new_net.load_state_dict(state_dict)
            self.net = new_net
            self.model_type = model_type
            self.model_path = path
            self.model_loaded = True
            self.model_type_var.set("CNN (卷积网络)" if model_type == "CNN" else "FNN (全连接网络)")
            self._update_model_status()
            messagebox.showinfo("成功", f"已加载 {model_type} 模型:\n{path}")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {e}")

    def _refresh_params(self):
        """刷新显示模型参数"""
        self.params_text.delete(1.0, tk.END)
        if not self.model_loaded:
            self.params_text.insert(tk.END, "模型未加载，无法查看参数。")
            return
        try:
            info = print_model_params(self.net)
            self.params_text.insert(tk.END, info)
        except Exception as e:
            self.params_text.insert(tk.END, f"获取参数失败: {e}")


# ========================================================================
#  入口
# ========================================================================
def main():
    root = tk.Tk()
    root.style = ttk.Style()
    try:
        root.style.theme_use("vista")
    except Exception:
        pass
    app = MNISTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
