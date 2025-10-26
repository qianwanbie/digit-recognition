import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# 1️⃣ 连接到 DagsHub 仓库
dagshub.init(repo_owner='qianwanbie', repo_name='digit-recognition', mlflow=True)

# 2️⃣ 加载数据
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 3️⃣ 设置 MLflow 实验名称
mlflow.set_experiment("RandomForestExperiment")

# 4️⃣ 定义训练 + 日志记录函数
def train_and_log_model(n_estimators, max_depth):
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # 记录参数、指标、模型
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"✅ Logged model: n_estimators={n_estimators}, max_depth={max_depth}, accuracy={accuracy:.4f}")

# 5️⃣ 运行两个实验
print("Training Model 1...")
train_and_log_model(20, 5)

print("\nTraining Model 2...")
train_and_log_model(100, 10)

print("\n✅ All experiments logged to DagsHub!")
