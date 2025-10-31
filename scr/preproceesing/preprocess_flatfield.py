# preprocess_flatfield.py
import os, csv
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

BASE = r"C:\Users\parth\Desktop\AI_track_finder"
DATASET_FLATFIELD = os.path.join(BASE, "dataset", "flatfield")
OUTPUT_DIR = os.path.join(BASE, "processed_data", "Flatfield")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "metadata_features.csv")

def load_and_preprocess(img_path, size=(512,512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def extract_noise_residual(img):
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    return img - denoised

def compute_metadata_features(img, file_path, scanner_id):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024.0
    pixels = img.flatten()
    mean_intensity = float(np.mean(pixels))
    std_intensity = float(np.std(pixels))
    skewness = float(skew(pixels))
    kurt = float(kurtosis(pixels))
    ent = float(entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6))
    edges = sobel(img)
    edge_density = float(np.mean(edges > 0.1))
    return {
        "file_name": os.path.basename(file_path),
        "main_class": "Flatfield",
        "resolution": "N/A",
        "class_label": scanner_id,
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(aspect_ratio),
        "file_size_kb": float(file_size_kb),
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def compute_flatfield_fingerprints(flatfield_dir, out_dir, csv_path):
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "file_name", "main_class", "resolution", "class_label",
            "width", "height", "aspect_ratio", "file_size_kb",
            "mean_intensity", "std_intensity", "skewness", "kurtosis",
            "entropy", "edge_density"
        ])
        writer.writeheader()

        for scanner_id in os.listdir(flatfield_dir):
            scanner_path = os.path.join(flatfield_dir, scanner_id)
            if not os.path.isdir(scanner_path):
                continue

            residuals = []
            files_processed = 0
            for file in os.listdir(scanner_path):
                if not file.lower().endswith(('.png','.tif','.tiff','.jpg','.jpeg')):
                    continue
                img_path = os.path.join(scanner_path, file)
                try:
                    img = load_and_preprocess(img_path)
                    residual = extract_noise_residual(img)
                    residuals.append(residual)
                    features = compute_metadata_features(img, img_path, scanner_id)
                    writer.writerow(features)
                    files_processed += 1
                except Exception as e:
                    print(f"⚠️ Skipping {img_path} due to error: {e}")
                    continue

            if residuals:
                fingerprint = np.mean(residuals, axis=0)
                np.save(os.path.join(out_dir, f"{scanner_id}_fingerprint.npy"), fingerprint)
                print(f"✅ Saved fingerprint for {scanner_id} ({files_processed} images)")

if __name__ == "__main__":
    compute_flatfield_fingerprints(DATASET_FLATFIELD, OUTPUT_DIR, CSV_PATH)
    print("🎯 Flatfield preprocessing + metadata feature extraction complete.")
