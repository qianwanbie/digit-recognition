import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import tkinter as tk
from tkinter import messagebox
import os

# ======== 模型定义（与训练时完全一致） ========
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
    # 转灰度
    image = image.convert('L')
    # 反相（黑底白字 → 白底黑字）
    image = ImageOps.invert(image)
    # 调整到 28x28
    image = image.resize((28, 28))
    # 转 numpy
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


# ======== 启动程序 ========
if __name__ == "__main__":
    model_path = "./training_results/final_model.pth"
    if not os.path.exists(model_path):
        messagebox.showerror("错误", f"模型文件未找到: {model_path}\n请先运行训练脚本生成模型。")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_path, device)
        DigitApp(model, device)
