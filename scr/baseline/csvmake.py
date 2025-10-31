import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# ================================
# Dataset folder
# ================================
DATASET_DIR = r"C:\Users\parth\Desktop\AI_track_finder\dataset\Official"
OUTPUT_CSV = r"C:\Users\parth\Desktop\AI_track_finder\features_dataset.csv"

# ================================
# Image preprocessing
# ================================
def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not load image: {img_path}")
        return None
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# ================================
# Compute features
# ================================
def compute_metadata_features(img, file_path):
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

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

# ================================
# Main CSV preparation
# ================================
all_features = []

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for root, _, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)
                img = load_and_preprocess(img_path)
                if img is not None:
                    features = compute_metadata_features(img, img_path)
                    features["label"] = class_name
                    all_features.append(features)
                    print(f"✅ Processed: {img_path}")

# Save to CSV
df = pd.DataFrame(all_features)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV of features saved at: {OUTPUT_CSV}")
