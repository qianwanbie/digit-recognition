# 使用国内可访问的 Python 3.8 slim 镜像
FROM dockerproxy.com/library/python:3.8-slim

# 设置工作目录
WORKDIR /app

# ✅ 先升级 pip、setuptools 和 wheel（防止 build_meta 错误）
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# ✅ 使用清华源安装依赖，加快国内速度
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认运行 app.py，如果要跑测试可以改为 ["pytest", "-v"]
CMD ["python", "app/app.py"]
