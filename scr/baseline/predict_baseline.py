# ============================================================
# Scanner Prediction Script (Educator Style)
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# ---------- MODEL PATHS ----------
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm.pkl")

# ---------- DATASET PATH ----------
# ✅ Windows-safe path to your dataset folder containing .tif images
DATASET_DIR = r"C:\Users\parth\Desktop\AI_track_finder\dataset\Official\EpsonV39-1\150"

# ============================================================
# IMAGE LOADING AND PREPROCESSING
# ============================================================
def load_and_preprocess(img_path, size=(512, 512)):
    print(f"[INFO] Loading image: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    print(f"[INFO] Image resized to: {size}")
    return img_resized

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def compute_metadata_features(img, file_path):
    print("[INFO] Computing metadata features...")
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    features_dict = {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density,
    }

    print("[INFO] Features computed successfully.")
    return features_dict

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_scanner(img_path, model_choice="rf"):
    print("[INFO] Loading scaler and model...")
    scaler = joblib.load(SCALER_PATH)
    if model_choice == "rf":
        model = joblib.load(RF_MODEL_PATH)
    else:
        model = joblib.load(SVM_MODEL_PATH)

    img = load_and_preprocess(img_path)
    features = compute_metadata_features(img, img_path)
    df_features = pd.DataFrame([features])
    X_scaled = scaler.transform(df_features)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    print("[INFO] Prediction completed successfully!")
    return pred, prob

# ============================================================
# AUTOMATIC TEST IMAGE SELECTION
# ============================================================
def get_test_image(dataset_dir):
    print(f"[INFO] Searching for a .tif image in: {dataset_dir}")
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".tif"):
                test_image = os.path.join(root, file)
                print(f"[INFO] Test image selected: {test_image}")
                return test_image
    raise FileNotFoundError("❌ No .tif images found in the dataset.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try:
        test_image = get_test_image(DATASET_DIR)
        pred, prob = predict_scanner(test_image, model_choice="rf")
        print(f"📌 Predicted Scanner: {pred}")
        print(f"📊 Class Probabilities: {prob}")
    except Exception as e:
        print(f"❌ Error: {e}")
