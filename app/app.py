import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from datetime import datetime
import subprocess
import argparse   # === 新增 ===

# 设置中文字体（修复字体警告）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 自定义数据集类
class HandwrittenDigitsDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.split = split
        
        # 加载数据
        labels_path = os.path.join(data_path, split, 'labels', f'{split}_labels.json')
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels_data = json.load(f)
        
        # 加载numpy数据
        X_path = os.path.join(data_path, split, 'labels', f'X_{split}.npy')
        y_path = os.path.join(data_path, split, 'labels', f'y_{split}.npy')
        
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx].reshape(28, 28).astype(np.float32) / 255.0
        label = self.y[idx]
        
        image_tensor = torch.FloatTensor(image).unsqueeze(0)
        label_tensor = torch.LongTensor([label]).squeeze()
        
        return image_tensor, label_tensor

# 2. 定义神经网络模型
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

# 3. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    save_dir = "./training_results"
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^12} | {'Val Acc':^10} | {'Time':^8}")
    print("-" * 70)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        epoch_time = time.time() - start_time
        
        # 记录历史
        history['train_loss'].append(float(avg_train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(avg_val_loss))
        history['val_acc'].append(float(val_acc))
        history['epoch_time'].append(float(epoch_time))
        
        # 打印进度
        print(f"{epoch+1:^6} | {avg_train_loss:^12.4f} | {train_acc:^10.2f}% | {avg_val_loss:^12.4f} | {val_acc:^10.2f}% | {epoch_time:^8.2f}s")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            model_path = os.path.join(save_dir, f'best_model_epoch{epoch+1}_acc{val_acc:.2f}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss
            }, model_path)
            
            print(f"✅ Saved best model: {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': avg_val_loss
    }, final_model_path)
    
    # 保存训练历史
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        history_serializable = history.copy()
        history_serializable['best_epoch'] = best_epoch
        history_serializable['best_val_acc'] = float(best_val_acc)
        history_serializable['training_time'] = sum(history['epoch_time'])
        history_serializable['total_epochs'] = epochs
        json.dump(history_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Total epochs: {epochs}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Training history saved: {history_path}")
    
    return history, save_dir

# （evaluate_model、plot_training_history 等函数保持完全不变，略）

# === 修改 main 函数 ===
def main():
    import subprocess
    parser = argparse.ArgumentParser(description="Train and evaluate handwritten digit model")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Dataset directory (default: ./dataset)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"📁 Using dataset directory: {args.data_dir}")   # === 新增提示 ===
    
    data_path = args.data_dir
    epochs = args.epochs

    # 自动从 DVC 拉取数据（如果不存在）
    if not os.path.exists(data_path):
        print(f"Dataset not found locally at {data_path}. Pulling from DVC...")
        try:
            subprocess.run(["dvc", "pull", data_path], check=True)
            print("✅ Dataset pulled successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to pull dataset from DVC.")
            return
    
    # 加载数据集
    print("Loading dataset...")
    train_dataset = HandwrittenDigitsDataset(data_path, 'train')
    val_dataset = HandwrittenDigitsDataset(data_path, 'val')
    test_dataset = HandwrittenDigitsDataset(data_path, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = DigitRecognizer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"\nTraining Configuration:")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training epochs: {epochs}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    history, save_dir = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)
    plot_training_history(history, save_dir, epochs)

    final_model_path = os.path.join(save_dir, 'final_model.pth')
    checkpoint = torch.load(final_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_accuracy, class_report, conf_matrix = evaluate_model(model, test_loader, device, save_dir)
    
    print(f"\n🎉 Training and evaluation completed!")
    print(f"Training epochs: {epochs}")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"All results saved in: {save_dir}")

if __name__ == "__main__":
    main()
