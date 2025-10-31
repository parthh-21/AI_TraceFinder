# ==========================================================
#  evaluate_cnn.py  
# ==========================================================

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "processed_data", "Official")  # Your .npy dataset
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model_from_npy.h5")

# ------------------ LOAD DATA ------------------
print("📥 Loading processed .npy files...")

X_list, y_list = [], []
label_map = {}
label_counter = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".npy"):
            # identify scanner/device
            parts = root.split(os.sep)
            for part in parts[::-1]:
                if part.lower().startswith(("canon", "epson", "hp")):
                    device_name = part
                    break
            else:
                continue

            if device_name not in label_map:
                label_map[device_name] = label_counter
                label_counter += 1

            label = label_map[device_name]
            file_path = os.path.join(root, file)
            data = np.load(file_path, allow_pickle=True)
            X_list.append(data)
            y_list.append(label)

print(f"✅ Loaded {len(X_list)} .npy files from {len(label_map)} scanners.")

# ------------------ PREPROCESS ------------------
X = np.array(X_list, dtype="float32") / 255.0
y = np.array(y_list)

if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)

num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

# ------------------ SPLIT DATA ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Dataset ready! X_test: {X_test.shape}, y_test: {y_test.shape}")

# ------------------ LOAD MODEL ------------------
print(f"📥 Loading CNN model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# ------------------ EVALUATE ------------------
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"📊 Test Accuracy: {acc*100:.2f}%")
print(f"📊 Test Loss: {loss:.4f}")
