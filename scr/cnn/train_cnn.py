# ==========================================================
#  train_cnn.py  — Educator Version (recursive .npy loader)
# ==========================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cnn/
DATA_DIR = os.path.join(BASE_DIR, "..", "processed_data", "Official")  # go one level up
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------ LOAD ALL .npy FILES RECURSIVELY ------------------
print("📥 Scanning dataset for .npy files (recursive)...")

X_list, y_list = [], []
label_map = {}  # maps device name → integer
label_counter = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".npy"):
            # find which scanner/device folder it belongs to
            parts = root.split(os.sep)
            for part in parts[::-1]:
                if part.lower().startswith(("canon", "epson", "hp")):  # device names
                    device_name = part
                    break
            else:
                continue

            if device_name not in label_map:
                label_map[device_name] = label_counter
                label_counter += 1

            label = label_map[device_name]
            file_path = os.path.join(root, file)

            try:
                data = np.load(file_path, allow_pickle=True)
                X_list.append(data)
                y_list.append(label)
            except Exception as e:
                print(f"⚠️ Could not load {file_path}: {e}")

print(f"✅ Loaded {len(X_list)} image files from {len(label_map)} scanners.")

if len(X_list) == 0:
    raise RuntimeError("❌ No .npy files found. Check dataset folder structure.")

# ------------------ PREPROCESSING ------------------
print("⚙️ Preprocessing data...")

X = np.array(X_list, dtype="float32") / 255.0
y = np.array(y_list)

if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)

num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✅ Dataset ready! X_train: {X_train.shape}, y_train: {y_train.shape}")

# ------------------ BUILD CNN ------------------
print("🧠 Building CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------ TRAIN MODEL ------------------
print("🚀 Training started...")

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ------------------ SAVE MODEL ------------------
model_path = os.path.join(MODEL_DIR, "cnn_model_from_npy.h5")
model.save(model_path)

print(f"\n✅ CNN model trained and saved successfully at: {model_path}")

# ------------------ EVALUATE MODEL ------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Final Test Accuracy: {acc * 100:.2f}%")
