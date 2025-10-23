# 使用官方 Python 3.8 slim 镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 升级 pip + 构建工具，防止 build_meta 错误
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目
COPY . .

# 默认命令为运行 pytest，适合 CI/CD
CMD ["pytest", "-v"]
