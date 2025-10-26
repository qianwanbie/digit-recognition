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
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 安装 DVC 带 HTTP 支持
RUN pip install --no-cache-dir "dvc[http]"

# 复制项目代码（不包含 .env）
COPY app /app/app
COPY dataset.dvc /app/dataset.dvc
COPY .dvc /app/.dvc

# 声明默认环境变量（可在运行容器时覆盖）
ENV DAGSHUB_USER=""
ENV DAGSHUB_TOKEN=""
ENV DAGSHUB_REPO="qianwanbie/digit-recognition"
ENV DAGSHUB_REMOTE="dagshub"
ENV DATA_PATH="./dataset"

# 启动时：
# 1️⃣ 根据环境变量配置 DVC
# 2️⃣ 拉取数据集
# 3️⃣ 启动应用
CMD sh -c '\
    dvc remote modify ${DAGSHUB_REMOTE} url https://dagshub.com/${DAGSHUB_REPO}.dvc && \
    dvc remote modify ${DAGSHUB_REMOTE} auth basic && \
    dvc remote modify ${DAGSHUB_REMOTE} user ${DAGSHUB_USER} && \
    dvc remote modify ${DAGSHUB_REMOTE} password ${DAGSHUB_TOKEN} && \
    dvc pull -r ${DAGSHUB_REMOTE} && \
    python app/app.py'
