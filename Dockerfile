# 使用 Python 3.11，兼容性更好
FROM python:3.11-slim

WORKDIR /app

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

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "dvc[http]"

COPY . .

# 环境变量（可在 docker run 时覆盖）
ENV DAGSHUB_USER=""
ENV DAGSHUB_TOKEN=""
ENV DAGSHUB_REPO="qianwanbie/digit-recognition"
ENV DAGSHUB_REMOTE="dagshub"
ENV DATA_PATH="./dataset"

CMD sh -c '\
    dvc remote add -d ${DAGSHUB_REMOTE} https://dagshub.com/${DAGSHUB_REPO}.dvc || true && \
    dvc remote modify ${DAGSHUB_REMOTE} auth basic && \
    dvc remote modify ${DAGSHUB_REMOTE} user ${DAGSHUB_USER} && \
    dvc remote modify ${DAGSHUB_REMOTE} password ${DAGSHUB_TOKEN} && \
    dvc pull -r ${DAGSHUB_REMOTE} && \
    python mlflow_tracking.py'
