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

# 设置 Dagshub token（直接在 Dockerfile 写死，不安全，仅限私有使用）
ENV DVC_HTTP_TOKEN=3f9abd6424a34a2fd3c6d2ff1fec7bd8fb938b74

# 确保远程名字正确
RUN dvc remote modify dagshub token $DVC_HTTP_TOKEN

# 拉取远程 dataset
RUN dvc pull -r dagshub

# 设置默认命令（可以运行测试或训练）
CMD ["python", "app/app.py"]
