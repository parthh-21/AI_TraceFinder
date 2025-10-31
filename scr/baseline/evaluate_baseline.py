# evaluate_baseline.py
import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# Paths
# ==========================
BASE = r"C:\Users\parth\Desktop\AI_track_finder"
CSV_PATH = os.path.join(BASE, "processed_data", "Flatfield", "metadata_features.csv")  # Change to Official if needed
MODELS_DIR = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# ==========================
# Load data
# ==========================
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
y = df["class_label"]

# ==========================
# Load scaler
# ==========================
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# ==========================
# Evaluation function
# ==========================
def evaluate_model(model_path, model_name):
    print(f"[INFO] Evaluating {model_name} model...")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_scaled)

    # Classification report
    print(f"\n=== {model_name} Evaluation ===")
    print(classification_report(y, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    save_path = os.path.join(RESULTS_DIR, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Confusion matrix saved to: {save_path}")
    plt.close()

# ==========================
# Run evaluation
# ==========================
evaluate_model(RF_MODEL_PATH, "Random Forest")
evaluate_model(SVM_MODEL_PATH, "SVM")
print("[INFO] Evaluation completed!")
