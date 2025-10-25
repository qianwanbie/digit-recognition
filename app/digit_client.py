import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import tkinter as tk
from tkinter import messagebox
import os
import subprocess
import sys
import time

# ======== 模型定义（必须与 app.py 保持一致） ========
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ======== 加载模型 ========
def load_model(model_path, device):
    model = DigitRecognizer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model


# ======== 图像预处理 ========
def preprocess_image(image):
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor


# ======== 主应用 ========
class DigitApp:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.root = tk.Tk()
        self.root.title("手写数字识别客户端")

        # 画布
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        # 按钮区
        frame = tk.Frame(self.root)
        frame.pack()
        tk.Button(frame, text="识别", command=self.predict_digit, width=10, height=2, bg="#4CAF50", fg="white").grid(row=0, column=0, padx=5)
        tk.Button(frame, text="清空", command=self.clear_canvas, width=10, height=2, bg="#F44336", fg="white").grid(row=0, column=1, padx=5)

        # 显示预测结果
        self.label = tk.Label(self.root, text="请在上方画布写下数字", font=("Helvetica", 16))
        self.label.pack(pady=10)

        # 内存绘图对象
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

        self.root.mainloop()

    def paint(self, event):
        x1, y1 = event.x - 8, event.y - 8
        x2, y2 = event.x + 8, event.y + 8
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill='white')
        self.label.config(text="请在上方画布写下数字")

    def predict_digit(self):
        img_tensor = preprocess_image(self.image).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted = torch.max(output, 1)
            digit = predicted.item()
        self.label.config(text=f"预测结果： {digit}")
        print(f"Predicted: {digit}")


# ======== 自动训练逻辑 ========
def ensure_model_exists():
    model_path = "./training_results/final_model.pth"
    if os.path.exists(model_path):
        print("✅ 已检测到训练好的模型，跳过训练。")
        return model_path

    print("⚠️ 未检测到模型文件，将自动启动训练程序 app.py ...")
    time.sleep(1)

    # 执行 app.py
    result = subprocess.run([sys.executable, "app.py"], check=False)
    if result.returncode != 0:
        messagebox.showerror("训练失败", "运行 app.py 时出现错误，请检查训练脚本。")
        sys.exit(1)

    if not os.path.exists(model_path):
        messagebox.showerror("错误", "训练完成后仍未生成模型文件，请检查 app.py。")
        sys.exit(1)

    print("✅ 模型训练完成！")
    return model_path


# ======== 主程序入口 ========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = ensure_model_exists()
    model = load_model(model_path, device)

    # 启动客户端
    DigitApp(model, device)
