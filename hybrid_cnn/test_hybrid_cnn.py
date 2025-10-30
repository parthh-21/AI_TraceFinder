# ============================================================
#  test_hybrid_cnn.py — Educator Version (with Parth’s paths)
# ============================================================

import os, pickle, cv2, numpy as np, tensorflow as tf
from skimage.feature import local_binary_pattern as sk_lbp

# ---------- PATHS ----------
BASE_DIR = r"C:\Users\parth\Desktop\AI_track_finder\dataset"
MODEL_PATH = os.path.join(BASE_DIR, "scanner_hybrid_final.keras")
FP_PATH = os.path.join(BASE_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(BASE_DIR, "fp_keys.npy")
ENC_PATH = os.path.join(BASE_DIR, "hybrid_label_encoder.pkl")
SCAL_PATH = os.path.join(BASE_DIR, "hybrid_feat_scaler.pkl")

# ---------- LOAD MODEL + COMPONENTS ----------
model = tf.keras.models.load_model(MODEL_PATH)
with open(FP_PATH, "rb") as f:
    scanner_fps = pickle.load(f)
fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
with open(ENC_PATH, "rb") as f:
    le = pickle.load(f)
with open(SCAL_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---------- HELPER FUNCTIONS ----------
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
    bins = np.linspace(0, r.max(), K+1)
    feats = [float(np.mean(mag[(r >= bins[i]) & (r < bins[i+1])])) for i in range(K)]
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

def extract_features(img_gray):
    img_resized = cv2.resize(img_gray, (256, 256))
    v_corr = [corr2d(img_resized, scanner_fps[k]) for k in fp_keys]
    v_fft = fft_radial_energy(img_resized, K=6)
    v_lbp = lbp_hist_safe(img_resized)
    feat = np.array(v_corr + v_fft + v_lbp).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    return img_resized.reshape(1, 256, 256, 1), feat_scaled

# ---------- TEST SINGLE IMAGE ----------
TEST_IMG_PATH = r"C:\Users\parth\Desktop\AI_track_finder\test_image.tif"

if not os.path.exists(TEST_IMG_PATH):
    raise FileNotFoundError(f"❌ Test image not found at: {TEST_IMG_PATH}")

img = cv2.imread(TEST_IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("⚠️ Could not read image. Check format (use .tif or .jpg).")

X_img, X_feat = extract_features(img)

preds = model.predict([X_img, X_feat])
top_idx = np.argsort(preds[0])[::-1][:3]

print("\n🔍 Top Predictions:")
for i in top_idx:
    print(f"{le.classes_[i]} — {preds[0][i]*100:.2f}%")
