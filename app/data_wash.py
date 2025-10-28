import os
import shutil
import json
import numpy as np
import random
from tqdm import tqdm

# =========================
# 配置路径
# =========================
DATA_ROOT = "./dataset"
OUTPUT_ROOT = "./dataset_cleaned"
SPLITS = ["train", "val", "test"]

# =========================
# 加载标签
# =========================
def load_labels(split_name):
    label_dir = os.path.join(DATA_ROOT, split_name, "labels")
    labels = {}
    
    # 检查 JSON 文件
    json_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]
    for jf in json_files:
        path = os.path.join(label_dir, jf)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 记录所有图片名
            for item in data:
                fname = item["filename"]
                labels[fname] = path
        except Exception as e:
            print(f"Error reading JSON {path}: {e}")

    # 检查 NPY 文件
    npy_files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    for nf in npy_files:
        path = os.path.join(label_dir, nf)
        try:
            arr = np.load(path, allow_pickle=True)
            # 根据长度生成对应图片名
            base_split = split_name
            for i in range(len(arr)):
                fname = f"{base_split}_{str(i).zfill(4)}.png"
                labels[fname] = path
        except Exception as e:
            print(f"Error reading NPY {path}: {e}")

    return labels  # {filename: label_path}

# =========================
# 数据清洗
# =========================
def clean_split(split_name, max_print=10):
    input_image_dir = os.path.join(DATA_ROOT, split_name, "images")
    output_image_dir = os.path.join(OUTPUT_ROOT, split_name, "images")
    output_label_dir = os.path.join(OUTPUT_ROOT, split_name, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    labels = load_labels(split_name)
    image_files = sorted(os.listdir(input_image_dir))

    kept_count = 0
    removed_samples = []

    print(f"\nProcessing {split_name} set...")
    for img_file in tqdm(image_files):
        img_path = os.path.join(input_image_dir, img_file)

        if img_file in labels:
            lbl_path = labels[img_file]
            # 尝试读取标签文件，能读取就保留
            ext = os.path.splitext(lbl_path)[1].lower()
            try:
                if ext == ".json":
                    _ = json.load(open(lbl_path, "r", encoding="utf-8"))
                elif ext == ".npy":
                    _ = np.load(lbl_path, allow_pickle=True)
                else:
                    raise ValueError("Unknown label type")
                # 复制图片和标签文件到 cleaned
                shutil.copy2(img_path, os.path.join(output_image_dir, img_file))
                shutil.copy2(lbl_path, os.path.join(output_label_dir, os.path.basename(lbl_path)))
                kept_count += 1
            except Exception as e:
                removed_samples.append((img_file, os.path.basename(lbl_path)))
                print(f"Error reading {lbl_path}: {e}")
        else:
            removed_samples.append((img_file, "NO LABEL"))

    print(f"Kept {kept_count}/{len(image_files)} samples for {split_name}")

    if removed_samples:
        print(f"\nSamples removed (up to {max_print} shown):")
        for img_file, lbl_file in random.sample(removed_samples, min(max_print, len(removed_samples))):
            print(f"- {img_file}, {lbl_file}")

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    if os.path.exists(OUTPUT_ROOT):
        print(f"Removing old cleaned dataset at {OUTPUT_ROOT}...")
        shutil.rmtree(OUTPUT_ROOT)

    for split in SPLITS:
        clean_split(split)

    print("\n✅ Data cleaning finished!")
