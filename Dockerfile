# 使用官方 Python 镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（兼容 Python 3.12-slim 和 OpenCV）
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
COPY . /app

# 升级 pip 并安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 设置默认命令
CMD ["python", "app.py"]
