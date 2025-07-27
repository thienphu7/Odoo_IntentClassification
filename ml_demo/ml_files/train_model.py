import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Dữ liệu huấn luyện đơn giản
texts = [
    "Where is my order?",
    "What is my shipment status?",
    "I want to track my order",
    "Can I become a seller?",
    "I want to join as a franchise",
    "How do I become a seller on your platform?",
]

labels = [
    "ORDER_STATUS",
    "ORDER_STATUS",
    "ORDER_STATUS",
    "FRANCHISE",
    "FRANCHISE",
    "FRANCHISE",
]

# Tạo pipeline: TF-IDF + Logistic Regression
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

# Huấn luyện mô hình
model.fit(texts, labels)

# Test nhanh
test_inputs = [
    "How do I become a seller?",
    "Where is my shipment?",
    "What is my order status?"
]
preds = model.predict(test_inputs)

for i, (inp, pred) in enumerate(zip(test_inputs, preds), 1):
    print(f"[{i}] Input: {inp} → Predicted: {pred}")

# Lưu mô hình vào model.pkl
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
joblib.dump(model, model_path)
print(f"\n Trained model saved to: {model_path}")