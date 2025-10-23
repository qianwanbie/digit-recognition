# 使用 Python 3.8 slim 镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖，保证科学计算库编译成功
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip、setuptools、wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制 requirements 并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认运行 app.py，可改成 pytest 测试
CMD ["python", "app/app.py"]
