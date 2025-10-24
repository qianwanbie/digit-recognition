# 使用官方 Python 3.12 镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（matplotlib、测试脚本需要）
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY app /app/app
COPY dataset /app/dataset
COPY requirements.txt /app/requirements.txt

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 设置默认命令（可以运行测试或训练）
CMD ["python", "app/app.py"]