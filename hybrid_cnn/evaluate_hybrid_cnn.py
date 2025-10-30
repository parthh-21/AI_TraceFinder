# ============================================================
#  evaluate_hybrid_cnn.py  — Educator Version (with Parth’s paths)
# ============================================================

import os, pickle, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- PATHS ----------
BASE_DIR = r"C:\Users\parth\Desktop\AI_track_finder\dataset"
ART_DIR  = BASE_DIR
MODEL_PATH = os.path.join(ART_DIR, "scanner_hybrid_final.keras")
HIST_PATH  = os.path.join(ART_DIR, "hybrid_training_history.pkl")
ENC_PATH   = os.path.join(ART_DIR, "hybrid_label_encoder.pkl")
SCAL_PATH  = os.path.join(ART_DIR, "hybrid_feat_scaler.pkl")
RES_PATH   = os.path.join(ART_DIR, "official_wiki_residuals.pkl")
FP_PATH    = os.path.join(ART_DIR, "scanner_fingerprints.pkl")
ORDER_NPY  = os.path.join(ART_DIR, "fp_keys.npy")

# ---------- LOAD ITEMS ----------
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)
with open(FP_PATH, "rb") as f:
    scanner_fps = pickle.load(f)
fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()

# ---------- FUNCTIONS ----------
def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a @ b) / denom) if denom != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K+1)
    feats = [float(np.mean(mag[(r >= bins[i]) & (r < bins[i+1])])) for i in range(K)]
    return feats

from skimage.feature import local_binary_pattern as sk_lbp
def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

# ---------- RECREATE DATASET ----------
X_img, X_feat, y = [], [], []
for dataset_name in ["Official", "wiki dataset"]:
    for scanner, dpi_dict in residuals_dict[dataset_name].items():
        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                X_img.append(np.expand_dims(res, -1))
                v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
                v_fft  = fft_radial_energy(res, K=6)
                v_lbp  = lbp_hist_safe(res)
                X_feat.append(v_corr + v_fft + v_lbp)
                y.append(scanner)

X_img = np.array(X_img, dtype=np.float32)
X_feat = np.array(X_feat, dtype=np.float32)
y = np.array(y)

# ---------- ENCODER + SCALER ----------
with open(ENC_PATH, "rb") as f:
    le = pickle.load(f)
with open(SCAL_PATH, "rb") as f:
    scaler = pickle.load(f)

y_true = le.transform(y)
X_feat = scaler.transform(X_feat)

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ---------- EVALUATE ----------
pred_probs = model.predict([X_img, X_feat])
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = y_true

# ---------- METRICS ----------
print("\n📊 Classification Report:\n")
print(classification_report(true_classes, pred_classes, target_names=le.classes_))

# ---------- CONFUSION MATRIX ----------
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix - Hybrid CNN Model")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.show()
