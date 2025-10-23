# test-app.py
import os
import torch
import numpy as np
import pytest
from app import HandwrittenDigitsDataset, DigitRecognizer, train_model, evaluate_model

# ==== 1. 测试数据集类 ====
def test_dataset_loading(tmp_path):
    """Test if HandwrittenDigitsDataset can load sample data correctly."""
    data_dir = tmp_path / "dataset" / "train" / "labels"
    os.makedirs(data_dir, exist_ok=True)

    # 创建伪造数据
    X = np.random.randint(0, 255, (5, 28, 28), dtype=np.uint8)
    y = np.random.randint(0, 10, (5,), dtype=np.int64)

    np.save(data_dir / "X_train.npy", X)
    np.save(data_dir / "y_train.npy", y)

    # 创建伪造 labels.json
    import json
    labels_json = [{"index": i, "label": int(y[i])} for i in range(len(y))]
    with open(data_dir / "train_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_json, f, ensure_ascii=False, indent=2)

    dataset = HandwrittenDigitsDataset(str(tmp_path / "dataset"), split="train")
    assert len(dataset) == 5
    sample_x, sample_y = dataset[0]
    assert sample_x.shape == (1, 28, 28)
    assert isinstance(sample_y.item(), int)

# ==== 2. 测试模型前向传播 ====
def test_model_forward():
    """Check that the model forward pass runs without errors."""
    model = DigitRecognizer()
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (4, 10), "Output shape should be (batch_size, num_classes)"

# ==== 3. 测试训练函数（快速运行 1 epoch） ====
def test_train_model(tmp_path):
    """Run one epoch of training on fake data to ensure training works."""
    X = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    dataset = [(X[i], y[i]) for i in range(8)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = DigitRecognizer()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history, save_dir = train_model(model, loader, loader, criterion, optimizer, "cpu", epochs=1)
    assert os.path.exists(save_dir)
    assert "train_loss" in history
    assert len(history["train_loss"]) == 1

# ==== 4. 测试评估函数 ====
def test_evaluate_model(tmp_path):
    """Ensure evaluation runs and saves outputs correctly."""
    model = DigitRecognizer()
    X = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    dataset = [(X[i], y[i]) for i in range(8)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    save_dir = tmp_path / "results"
    os.makedirs(save_dir, exist_ok=True)

    acc, report, cm = evaluate_model(model, loader, "cpu", str(save_dir))
    assert isinstance(acc, float)
    assert os.path.exists(save_dir / "evaluation_results.json")
    assert os.path.exists(save_dir / "confusion_matrix.png")
