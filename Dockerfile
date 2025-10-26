# 使用官方 Python 3.12 镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
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

# 安装 Python 依赖
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件和 DVC 配置文件
COPY app /app/app
COPY dataset.dvc /app/dataset.dvc
COPY .dvc /app/.dvc

# 安装 DVC 带 http 支持
RUN pip install --no-cache-dir "dvc[http]"

# 硬编码 token 到 .dvc/config（保留运行时可用）
RUN mkdir -p /app/.dvc && \
    echo "[remote \"dagshub\"]" > /app/.dvc/config && \
    echo "    url = https://dagshub.com/qianwanbie/digit-recognition.dvc" >> /app/.dvc/config && \
    echo "    token = 3f5d8568e630e550a8e294e6acbe0eeb4d278b34" >> /app/.dvc/config

# 设置默认命令：先拉取 DVC dataset，再运行程序
CMD ["sh", "-c", "dvc pull -r dagshub && python app/app.py"]
