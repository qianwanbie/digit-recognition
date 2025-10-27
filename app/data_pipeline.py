# =========================================
# data_pipeline.py — 生成 dataset_cleaned/
# =========================================
import os
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import shutil

# ---------- 路径设置 ----------
SOURCE_DIR = Path("dataset")         # 原始数据
OUTPUT_DIR = Path("dataset_cleaned") # 清洗后输出

# ---------- 工具函数 ----------
def ensure_dir(path: Path):
    os.makedirs(path, exist_ok=True)

def clean_data(X, y):
    """简单清洗：去除NaN并标准化"""
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def process_split(split):
    print(f"🧹 处理 {split} 数据集...")

    src_labels = SOURCE_DIR / split / "labels"
    dst_labels = OUTPUT_DIR / split / "labels"
    ensure_dir(dst_labels)

    X = np.load(src_labels / f"X_{split}.npy")
    y = np.load(src_labels / f"y_{split}.npy")
    with open(src_labels / f"{split}_labels.json", "r", encoding="utf-8") as f:
        label_info = json.load(f)

    print(f"原始形状: {X.shape}")
    X_clean, y_clean = clean_data(X, y)
    print(f"清洗后形状: {X_clean.shape}")

    np.save(dst_labels / f"X_{split}.npy", X_clean)
    np.save(dst_labels / f"y_{split}.npy", y_clean)
    with open(dst_labels / f"{split}_labels.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=4, ensure_ascii=False)

    # 复制 images 文件夹
    src_images = SOURCE_DIR / split / "images"
    dst_images = OUTPUT_DIR / split / "images"
    if src_images.exists():
        shutil.copytree(src_images, dst_images, dirs_exist_ok=True)

# ---------- 主程序 ----------
def main():
    print("=== 🚀 启动数据清洗 ===")
    ensure_dir(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        process_split(split)

    print(f"\n✅ 数据清洗完成！新数据位于: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
