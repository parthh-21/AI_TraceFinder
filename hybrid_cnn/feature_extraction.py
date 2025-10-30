import os
import pickle
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy import ndimage
from scipy.fft import fft2, fftshift

# ---------------------------
# 🔧 Paths (adjusted for your system)
# ---------------------------
BASE_DIR = r"C:\Users\parth\Desktop\AI_track_finder\dataset"
FLATFIELD_RESIDUALS_PATH = os.path.join(BASE_DIR, "flatfield_residuals.pkl")
FP_OUT_PATH = os.path.join(BASE_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(BASE_DIR, "fp_keys.npy")
RES_PATH = os.path.join(BASE_DIR, "official_wiki_residuals.pkl")
FEATURES_OUT = os.path.join(BASE_DIR, "features.pkl")
ENHANCED_OUT = os.path.join(BASE_DIR, "enhanced_features.pkl")

# ---------------------------
# 1️⃣ Compute scanner fingerprints
# ---------------------------
with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
    flatfield_residuals = pickle.load(f)

scanner_fingerprints = {}
print("🔄 Computing fingerprints from Flatfield dataset...")
for scanner, residuals in flatfield_residuals.items():
    if not residuals:
        continue
    stack = np.stack(residuals, axis=0)
    fingerprint = np.mean(stack, axis=0)
    scanner_fingerprints[scanner] = fingerprint

# Save fingerprints + order
with open(FP_OUT_PATH, "wb") as f:
    pickle.dump(scanner_fingerprints, f)
fp_keys = sorted(scanner_fingerprints.keys())
np.save(ORDER_NPY, np.array(fp_keys))
print(f"✅ Saved {len(scanner_fingerprints)} fingerprints and fp_keys.npy")

# ---------------------------
# 2️⃣ PRNU Correlation Features
# ---------------------------
def corr2d(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0

with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

features, labels = [], []
for dataset_name in ["Official", "wiki dataset"]:
    print(f"🔄 Computing PRNU features for {dataset_name} ...")
    for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                vec = [corr2d(res, scanner_fingerprints[k]) for k in fp_keys]
                features.append(vec)
                labels.append(scanner)

with open(FEATURES_OUT, "wb") as f:
    pickle.dump({"features": features, "labels": labels}, f)
print(f"✅ Saved PRNU features — Shape: {len(features)} × {len(features[0])}")

# ---------------------------
# 3️⃣ Enhanced Features (FFT + LBP + Texture)
# ---------------------------
def extract_enhanced_features(residual):
    # FFT energy
    fft_img = np.abs(fft2(residual))
    fft_img = fftshift(fft_img)
    h, w = fft_img.shape
    ch, cw = h // 2, w // 2
    low_freq = np.mean(fft_img[ch-20:ch+20, cw-20:cw+20])
    mid_freq = np.mean(fft_img[ch-60:ch+60, cw-60:cw+60]) - low_freq
    high_freq = np.mean(fft_img) - low_freq - mid_freq

    # LBP histogram
    lbp = local_binary_pattern(residual, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0,25), density=True)

    # Gradient-based texture
    grad_x = ndimage.sobel(residual, axis=1)
    grad_y = ndimage.sobel(residual, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    texture_feats = [np.std(residual), np.mean(np.abs(residual)),
                     np.std(grad_mag), np.mean(grad_mag)]

    return [low_freq, mid_freq, high_freq] + lbp_hist.tolist() + texture_feats

enhanced_features, enhanced_labels = [], []
for dataset_name in ["Official", "wiki dataset"]:
    print(f"🔄 Extracting enhanced features for {dataset_name} ...")
    for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                feat = extract_enhanced_features(res)
                enhanced_features.append(feat)
                enhanced_labels.append(scanner)

with open(ENHANCED_OUT, "wb") as f:
    pickle.dump({"features": enhanced_features, "labels": enhanced_labels}, f)
print(f"✅ Saved enhanced features — Shape: {len(enhanced_features)} × {len(enhanced_features[0])}")
print("🎯 Feature extraction complete!")
