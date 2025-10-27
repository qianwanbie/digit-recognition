一、项目概述 / Project Overview

本项目基于 PyTorch + MLflow + DagsHub，实现从数据清洗、模型训练、实验记录到模型验证和客户端预测的完整 MLOps 流程。
This project is built on PyTorch, MLflow, and DagsHub, implementing a full MLOps workflow from data preprocessing, model training, experiment logging to model validation and client prediction.

📘 本文件用于指导如何部署本项目，使数据处理、训练、实验追踪、模型验证、Docker 构建和 CI 测试流程可顺利运行。
This document explains how to deploy the project so that data preprocessing, training, experiment tracking, model validation, Docker builds, and CI testing workflows run smoothly.

⚙️ 二、环境要求 / Environment Requirements
组件 / Component	版本要求 / Required Version	说明 / Description
Python	3.10+	    主环境语言 / Main runtime
pip	23.0+	        包管理工具 / Package manager
Git	Latest	        项目版本控制 / Version control
Docker	Optional	容器化部署 / Container deployment
MLflow	Latest	    模型追踪与管理 / Model tracking
DagsHub	Account required	云端实验记录 / Remote MLflow hosting

🧰 三、环境配置步骤 / Environment Setup Steps
1️⃣ 克隆项目 / Clone the Repository
git clone https://github.com/qianwanbie/digit-recognition.git

2️⃣ 创建虚拟环境 / Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows

3️⃣ 安装依赖 / Install Dependencies
pip install -r requirements.txt

4️⃣ 配置环境变量 / Set Up Environment Variables

创建 .env 文件，并填写 MLflow / DagsHub 凭证：
Create a .env file and fill in MLflow/DagsHub credentials:

MLFLOW_TRACKING_URI=https://dagshub.com/ qianwanbie/ digit-recognition.mlflow
MLFLOW_TRACKING_USERNAME=${{ secrets.DAGSHUB_USER }}
MLFLOW_TRACKING_PASSWORD=${{ secrets.DAGSHUB_TOKEN }}

🧹 四、数据清洗与预处理 / Data Cleaning & Preprocessing

项目中 app/data_pipeline.py 用于对原始数据集进行清洗和预处理，生成训练和测试所需格式的数据。
The app/data_pipeline.py script is used to clean and preprocess raw datasets to generate the format required for training and testing.

python app/data_pipeline.py


处理后的数据默认保存至 dataset_cleaned 目录。
The processed data is saved to the dataset_cleaned directory by default.

🧠 五、模型训练与实验记录 / Model Training and Experiment Logging

运行训练脚本，同时使用 MLflow 记录实验：

python mlflow_tracking.py


mlflow_tracking.py 会调用 app.py 中的训练函数执行模型训练。
它会自动记录实验结果、超参数、精度等信息，并上传至 DagsHub。
mlflow_tracking.py calls the training functions in app.py to train the model.
It automatically logs experiment metrics, hyperparameters, and performance, and uploads them to DagsHub.

训练模型文件会保存到 training_results/。
Trained model files are saved in training_results/.

🧪 六、模型验证与预测测试 / Model Validation & Prediction Testing

使用测试脚本对训练好的模型进行验证或预测：

python test_app.py


test_app.py 会加载训练好的模型，对测试数据进行预测，并生成评估指标（如 accuracy、confusion matrix 等）。
输出结果会保存至 training_results/ 文件夹，以便分析和对比。
test_app.py loads the trained model, runs predictions on the test dataset, and generates evaluation metrics (e.g., accuracy, confusion matrix).
The results are saved in training_results/ for analysis and comparison.

💻 七、客户端 / Client Application

项目中 app/digit_client.py 实现了客户端功能，可接收用户输入（例如手写数字图片）并返回模型预测结果。
The app/digit_client.py script implements the client, which can accept user input (e.g., handwritten digit images) and return model predictions.

python app/digit_client.py


客户端使用训练好的模型进行实时预测，并可选择保存或展示结果。
The client uses the trained model for real-time prediction and optionally saves or displays the results.

🐳 八、Docker 部署 / Docker Deployment
1️⃣ 构建 Docker 镜像 / Build Docker Image
docker build -t final-project:latest .

2️⃣ 启动 Docker 容器 / Run Docker Container
docker run -d final-project:latest


注意：由于项目当前不包含 Web 服务，本容器仅用于封装训练/验证环境。
Note: The project does not include a web service; the container is used only to encapsulate the training/validation environment.

3️⃣ Docker 自动构建 / Docker Automated Build

如果配置了 docker-build.yml，可通过 GitHub Actions 自动构建镜像：

# 推送到远程仓库触发 CI/CD
git push origin main


docker-build.yml 会在推送时自动执行 Docker 构建，并生成可用镜像。
docker-build.yml automatically triggers Docker image build when you push code to the repository.

🔧 九、Python 自动化测试 / Python Automated Testing

使用 GitHub Actions 配置的 python-tests.yml 进行 CI 测试：

# 本地可运行测试
pytest test_app.py


python-tests.yml 配置了自动执行 pytest 测试，用于验证模型训练、数据加载、前向推理和评估函数是否正常。
python-tests.yml is configured to run pytest automatically to verify that model training, data loading, forward inference, and evaluation functions work correctly.

🧱 十、项目目录结构 / Project Structure
final-project/
│
├── app/                        # 模型、训练/评估函数及工具 / Model, training/evaluation functions & utilities
│   ├── app.py                  # 模型定义与训练/评估函数 / Model definition & training/evaluation
│   ├── data_pipeline.py        # 数据清洗/预处理脚本 / Data cleaning & preprocessing
│   └── digit_client.py         # 客户端实现脚本 / Client application
├── data/                        # 数据集 / Dataset
├── training_results/            # 训练输出 / Training outputs
    # 保存训练好的模型 / Trained models               # 测试与验证输出 / Prediction & evaluation outputs
├── mlflow_tracking.py          # 训练与 MLflow 追踪脚本 / Training & MLflow tracking
├── test_app.py                 # 模型验证/预测测试脚本 / Model validation & prediction
├── requirements.txt            # 依赖文件 / Dependencies
├── Dockerfile                  # Docker 镜像构建配置 / Docker image build configuration
├── docker-build.yml            # Docker 自动构建/部署工作流 / Docker automated build/deployment workflow
├── python-tests.yml            # GitHub Actions / CI 测试配置 / CI workflow for Python tests
├── .env                        # 环境变量 / Environment variables
└── DEPLOYMENT.md               # 部署说明 / Deployment guide