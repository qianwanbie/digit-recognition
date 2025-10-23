FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 运行测试（或可改为 app.py）
CMD ["python", "app/app.py"]
