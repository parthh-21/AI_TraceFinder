# app.py
# ==========================================================
# AI TraceFinder - Educator-style Streamlit UI
# - Sidebar nav: Home, Dataset Overview, Feature Visualization,
#   Model Training & Evaluation, Live Prediction, About
# - Auto-detects artifacts under dataset/
# - Uses hybrid scanner model + patch-level tamper SVM fallback
# ==========================================================

import os, sys, math, json, pickle, tempfile
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import cv2, pywt
import tensorflow as tf
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# ---------------------------
# Config / paths
# ---------------------------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "dataset"
TAMpered_DIR = DATA_DIR / "Tampered images"
TAMPER_MODELS_DIR = TAMpered_DIR / "tamper_models"

# Scanner artifacts (now in tamper_models)
SCN_MODEL_PATH  = TAMPER_MODELS_DIR / "scanner_hybrid.keras"
SCN_LE_PATH     = TAMPER_MODELS_DIR / "hybrid_label_encoder.pkl"
SCN_SCALER_PATH = TAMPER_MODELS_DIR / "hybrid_feat_scaler.pkl"
SCN_FP_PATH     = TAMPER_MODELS_DIR / "scanner_fingerprints.pkl"
SCN_FP_KEYS     = TAMPER_MODELS_DIR / "fp_keys.npy"
HYB_HISTORY     = TAMPER_MODELS_DIR / "hybrid_training_history.pkl"

# Image-level tamper (optional)
IMG_SCALER_PATH = TAMPER_MODELS_DIR / "image_scaler.pkl"
IMG_CLF_PATH    = TAMPER_MODELS_DIR / "image_svm_sig.pkl"
IMG_THR_JSON    = TAMPER_MODELS_DIR / "image_thresholds.json"

# Patch-level tamper fallback
TP_SCALER_PATH  = TAMPER_MODELS_DIR / "patch_scaler.pkl"
TP_CLF_PATH     = TAMPER_MODELS_DIR / "patch_svm_sig_calibrated.pkl"
TP_THR_JSON     = TAMPER_MODELS_DIR / "thresholds_patch.json"

# CSV manifests (optional)
WIKI_CSV = DATA_DIR / "wiki dataset" / "wiki_manifest.csv"
FLAT_CSV = DATA_DIR / "flatfield" / "flatfield_manifest.csv"
TAMPER_MANIFEST = TAMPER_MODELS_DIR / "tamper_manifest.csv"

# Constants
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16

# ---------------------------
# Utility: safe loaders & checks
# ---------------------------
def exists(p):
    return p is not None and Path(p).exists()

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_np(path):
    return np.load(path, allow_pickle=True)

@st.cache_resource
def load_tf(path):
    return tf.keras.models.load_model(str(path), compile=False)

# ---------------------------
# Auto-detect available artifacts
# ---------------------------
AVAILABLE = {}
AVAILABLE["scanner_model"] = exists(SCN_MODEL_PATH)
AVAILABLE["scanner_files"] = all(map(exists, [SCN_LE_PATH, SCN_SCALER_PATH, SCN_FP_PATH, SCN_FP_KEYS]))
AVAILABLE["patch_model"] = exists(TP_SCALER_PATH) and exists(TP_CLF_PATH)
AVAILABLE["image_model"] = exists(IMG_SCALER_PATH) and exists(IMG_CLF_PATH)
AVAILABLE["hyb_history"] = exists(HYB_HISTORY)
AVAILABLE["tamper_manifest"] = exists(TAMPER_MANIFEST)

# ---------------------------
# Load artifacts (if present)
# ---------------------------
scanner_model = scanner_le = scanner_scaler = scanner_fps = fp_keys = None
if AVAILABLE["scanner_model"] and AVAILABLE["scanner_files"]:
    try:
        scanner_model = load_tf(SCN_MODEL_PATH)
        scanner_le = load_pickle(SCN_LE_PATH)
        scanner_scaler = load_pickle(SCN_SCALER_PATH)
        scanner_fps = load_pickle(SCN_FP_PATH)
        fp_keys = load_np(SCN_FP_KEYS).tolist()
    except Exception as e:
        st.warning(f"Warning loading scanner artifacts: {e}")
        scanner_model = scanner_le = scanner_scaler = scanner_fps = fp_keys = None
        AVAILABLE["scanner_model"] = False

patch_scaler = patch_clf = patch_thr = None
if AVAILABLE["patch_model"]:
    try:
        patch_scaler = load_pickle(TP_SCALER_PATH)
        patch_clf = load_pickle(TP_CLF_PATH)
        if exists(TP_THR_JSON): patch_thr = load_json(TP_THR_JSON)
    except Exception as e:
        st.warning(f"Warning loading patch artifacts: {e}")
        patch_scaler = patch_clf = patch_thr = None
        AVAILABLE["patch_model"] = False

img_scaler = img_clf = img_thr = None
if AVAILABLE["image_model"]:
    try:
        img_scaler = load_pickle(IMG_SCALER_PATH)
        img_clf = load_pickle(IMG_CLF_PATH)
        if exists(IMG_THR_JSON): img_thr = load_json(IMG_THR_JSON)
    except Exception as e:
        st.warning(f"Warning loading image-level artifacts: {e}")
        img_scaler = img_clf = img_thr = None
        AVAILABLE["image_model"] = False

hyb_history = None
if AVAILABLE["hyb_history"]:
    try:
        hyb_history = load_pickle(HYB_HISTORY)
    except Exception:
        hyb_history = None

# ---------------------------
# Feature helpers (NumPy2 safe)
# ---------------------------
def preprocess_residual(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    return (img - den).astype(np.float32)

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h,:w]; r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    feats = []
    for i in range(K):
        m = (r>=bins[i]) & (r<bins[i+1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    if rng < 1e-12:
        g = np.zeros_like(img, dtype=np.float32)
    else:
        g = (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins+1), density=True)
    return hist.astype(np.float32)

# ---------------------------
# Scanner feature builder (safe)
# ---------------------------
def make_scanner_feats(res):
    if not (scanner_fps and fp_keys):
        raise RuntimeError("Scanner fingerprints or keys missing.")
    if len(fp_keys) != 11:
        raise RuntimeError(f"fp_keys length {len(fp_keys)} != 11")
    v_corr = []
    for k in fp_keys:
        if k not in scanner_fps:
            raise RuntimeError(f"fingerprint key '{k}' missing in scanner_fps")
        v_corr.append(corr2d(res, scanner_fps[k]))
    v_fft = fft_radial_energy(res, 6)
    v_lbp = lbp_hist_safe(res, 8, 1.0).tolist()
    # ensure lengths
    if len(v_fft) != 6: v_fft = (v_fft + [0.0]*6)[:6]
    if len(v_lbp) != 10:
        if len(v_lbp) > 10: v_lbp = v_lbp[:10]
        else: v_lbp = v_lbp + [0.0]*(10-len(v_lbp))
    feat = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1,-1)
    return scanner_scaler.transform(feat)

def predict_scanner(image_path):
    res = preprocess_residual(image_path)
    x_img = np.expand_dims(res, axis=(0,-1))
    x_ft = make_scanner_feats(res)
    prob = scanner_model.predict([x_img, x_ft], verbose=0).ravel()
    idx = int(np.argmax(prob))
    label = scanner_le.classes_[idx]
    conf = float(prob[idx]*100.0)
    return label, conf

# ---------------------------
# Patch tamper utilities
# ---------------------------
def residual_stats(img):
    return np.array([float(img.mean()), float(img.std()), float(np.mean(np.abs(img)))], dtype=np.float32)

def fft_resample_feats(img):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h,:w]; r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    rmax = r.max() + 1e-6
    b1 = (r>=0.25*rmax) & (r<0.35*rmax)
    b2 = (r>=0.35*rmax) & (r<0.5*rmax)
    e1 = float(mag[b1].mean() if b1.any() else 0.0)
    e2 = float(mag[b2].mean() if b2.any() else 0.0)
    ratio = float(e2/(e1+1e-8))
    return np.array([e1,e2,ratio], dtype=np.float32)

def make_patch_feat_22(patch):
    lbp = np.asarray(lbp_hist_safe(patch,8,1.0), np.float32)
    fft6 = np.asarray(fft_radial_energy(patch,6), np.float32)
    res3 = residual_stats(patch)
    rsp3 = fft_resample_feats(patch)
    return np.concatenate([lbp, fft6, res3, rsp3], 0)

def extract_patches(res, patch=PATCH, stride=STRIDE, limit=MAX_PATCHES, seed=1234):
    H,W = res.shape
    ys = list(range(0, H-patch+1, stride)); xs = list(range(0, W-patch+1, stride))
    coords = [(y,x) for y in ys for x in xs]
    rng = np.random.RandomState(seed)
    rng.shuffle(coords)
    coords = coords[:min(limit, len(coords))]
    return [res[y:y+patch, x:x+patch] for y,x in coords]

def image_score_topk(probs, frac=0.3):
    n = len(probs); k = max(1, int(math.ceil(frac*n)))
    return float(np.mean(np.sort(np.asarray(probs))[-k:]))

def choose_thr_patch(domain):
    if patch_thr is None: return 0.5
    return patch_thr.get("by_domain", {}).get(domain, patch_thr.get("global", 0.5))

def infer_patch_tamper(image_path, frac=0.3, local_gate=0.85, min_hits=2, seed=1234):
    if not AVAILABLE["patch_model"]:
        return {"prob_tampered": 0.0, "tamper_label":"Unknown", "hits":0, "confidence":0.0}
    res = preprocess_residual(image_path)
    patches = extract_patches(res, limit=MAX_PATCHES, seed=seed)
    feats = np.stack([make_patch_feat_22(p) for p in patches], 0)
    feats = patch_scaler.transform(feats)
    p_patch = patch_clf.predict_proba(feats)[:,1]
    p_img = image_score_topk(p_patch, frac=frac)
    dom = "tamper_dir"
    thr = choose_thr_patch(dom)
    hits = int((p_patch >= local_gate).sum())
    tampered = int((p_img >= thr) and (hits >= min_hits))
    conf = float((p_img if tampered else 1.0 - p_img) * 100.0)
    return {"prob_tampered": p_img, "tamper_label": "Tampered" if tampered else "Clean", "hits": hits, "confidence": conf}

# ---------------------------
# Image-level tamper (18-D) - optional
# ---------------------------
def contrast_stat_single(img):
    return np.asarray([float(img.std()), float(np.mean(np.abs(img)-np.mean(img)))], dtype=np.float32)

def image_patch_feat_18(patch):
    lbp10 = np.asarray(lbp_hist_safe(patch,8,1.0), np.float32)
    fft6 = np.asarray(fft_radial_energy(patch,6), np.float32)
    cs2 = contrast_stat_single(patch)
    return np.concatenate([lbp10, fft6, cs2], 0)

def make_image_feat_18(res, seed=1234):
    patches = extract_patches(res, limit=MAX_PATCHES, seed=seed)
    if not patches:
        return np.zeros((1,18), dtype=np.float32)
    feats = np.stack([image_patch_feat_18(p) for p in patches], 0)
    return feats.mean(axis=0, keepdims=True).astype(np.float32)

def infer_image_tamper(image_path):
    if not AVAILABLE["image_model"]:
        return {"prob_tampered": 0.0, "tamper_label":"Unknown", "confidence":0.0}
    res = preprocess_residual(image_path)
    feat = make_image_feat_18(res)
    feat = img_scaler.transform(feat)
    p = float(img_clf.predict_proba(feat)[:,1][0])
    thr = img_thr.get("global", 0.5) if img_thr else 0.5
    tampered = int(p >= thr)
    conf = float((p if tampered else 1.0-p)*100.0)
    return {"prob_tampered": p, "tamper_label": "Tampered" if tampered else "Clean", "confidence": conf}

# ---------------------------
# UI: Sidebar nav
# ---------------------------
st.set_page_config(page_title="TraceFinder - Forensic Scanner ID", layout="wide")
st.sidebar.title("TraceFinder")
st.sidebar.markdown("Forensic Scanner Identification — educator-style UI")

page = st.sidebar.radio("Navigation", ["Home", "Dataset Overview", "Feature Visualization", "Model Training & Evaluation", "Live Prediction", "About"])

# Quick status box
with st.sidebar.expander("Model status", expanded=True):
    st.write("Scanner model:", "Loaded" if AVAILABLE["scanner_model"] else "Missing")
    st.write("Patch tamper:", "Loaded" if AVAILABLE["patch_model"] else "Missing")
    st.write("Image tamper:", "Loaded" if AVAILABLE["image_model"] else "Missing")

# ---------------------------
# Pages
# ---------------------------
if page == "Home":
    st.title("TraceFinder - Forensic Scanner Identification")
    
    st.markdown("""
    Welcome to **AI TraceFinder**. This application is an educational tool designed to explore 
    two key areas of digital image forensics:

    1.  **Source Scanner Identification:** Identifying the specific scanner model used to create an image.
    2.  **Image Tamper Detection:** Determining if an image has been digitally altered.

    Use the navigation panel on the left to explore the dataset, visualize image features, 
    or run a live prediction.
    """)
    st.markdown("---")
    
    # Quick stats
    total_images = 0
    scanner_classes = 0
    dpi_levels = "N/A"
    if DATA_DIR.exists():
        for ext in ("*.tif","*.tiff","*.png","*.jpg","*.jpeg"):
            total_images += len(list(DATA_DIR.rglob(ext)))
    if scanner_le is not None:
        scanner_classes = len(getattr(scanner_le, "classes_", []))
    st.subheader("Quick Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total images", total_images)
    col2.metric("Scanner classes", scanner_classes)
    col3.metric("Example DPI levels", "150,300,600")
    st.markdown("---")
    st.image(str(BASE / "Screenshot 2025-10-29 181608.png"), caption="   ", use_column_width=True)

elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write("Folders in dataset:")
    folders = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
    st.write(folders)
    st.write("---")
    # Show first few images from Tampered images (if present)
    if TAMpered_DIR.exists():
        sample = list(TAMpered_DIR.rglob("*.tif"))[:8] + list(TAMpered_DIR.rglob("*.png"))[:8] + list(TAMpered_DIR.rglob("*.jpg"))[:8]
        sample = [p for p in sample][:8]
        cols = st.columns(4)
        for i,p in enumerate(sample):
            with cols[i%4]:
                try:
                    img = Image.open(p).convert("RGB")
                    st.image(img.resize((200,200)), caption=p.name)
                except Exception:
                    st.write(p.name)

elif page == "Feature Visualization":
    st.title("Feature Visualization")
    st.write("Visualize residuals, LBP histograms, FFT radial energy for a selected image.")
    img_path = st.file_uploader("Upload image to visualize (or enter path below)", type=["png","jpg","jpeg","tif","tiff"])
    manual = st.text_input("Or enter full path to an image:", "")
    target = None
    if img_path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_path.name)[1])
        tmp.write(img_path.read()); tmp.close(); target = tmp.name
    elif manual:
        target = manual.strip()
    if target:
        try:
            res = preprocess_residual(target)
            st.subheader("Residual (wavelet denoised)")
            plt.figure(figsize=(6,4)); plt.imshow(res, cmap="gray"); plt.axis("off")
            st.pyplot(plt.gcf()); plt.clf()

            st.subheader("LBP histogram (P=8,R=1)")
            lbp = lbp_hist_safe(res,8,1.0)
            fig,ax = plt.subplots(figsize=(6,3))
            ax.bar(np.arange(len(lbp)), lbp); ax.set_xlabel("LBP bin"); ax.set_ylabel("Density")
            st.pyplot(fig); plt.clf()

            st.subheader("FFT radial energy (6 bins)")
            f6 = fft_radial_energy(res,6)
            fig,ax = plt.subplots(figsize=(6,3)); ax.plot(f6, marker="o"); ax.set_xlabel("Bin"); ax.set_ylabel("Energy")
            st.pyplot(fig); plt.clf()
        except Exception as e:
            st.error(f"Error visualizing image: {e}")

elif page == "Model Training & Evaluation":
    st.title("Model Training & Evaluation")
    st.write("View training history and metrics if available.")
    if hyb_history:
        st.write("Hybrid training history keys:", list(hyb_history.keys()))
        # if it contains 'loss' and 'val_loss' arrays, plot
        if "loss" in hyb_history:
            fig,ax = plt.subplots()
            ax.plot(hyb_history.get("loss", []), label="loss")
            ax.plot(hyb_history.get("val_loss", []), label="val_loss")
            ax.legend(); st.pyplot(fig); plt.clf()
    else:
        st.info("No hybrid_training_history.pkl found in dataset/")

    st.write("---")
    st.write("Patch model status:")
    st.write("Patch scaler:", "Loaded" if AVAILABLE["patch_model"] else "Missing")
    if AVAILABLE["patch_model"]:
        st.write("You can evaluate patch SVM externally using testing.py. Manifest:", str(TAMPER_MANIFEST) if TAMPER_MANIFEST.exists() else "No manifest found")

elif page == "Live Prediction":
    st.title("Live Prediction — Scanner & Tamper")
    uploaded = st.file_uploader("Upload an image (tif/png/jpg)", type=["tif","tiff","png","jpg","jpeg"])
    run = st.button("Run", disabled=(uploaded is None))
    if uploaded and run:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read()); tmp.close()
        try:
            # scanner (if available)
            if AVAILABLE["scanner_model"]:
                try:
                    s_label, s_conf = predict_scanner(tmp.name)
                except Exception as e:
                    s_label, s_conf = "Error", 0.0
                    st.error(f"Scanner prediction error: {e}")
            else:
                s_label, s_conf = "Unavailable", 0.0

            # tamper: prefer image-level if available, else patch fallback
            if AVAILABLE["image_model"]:
                t_res = infer_image_tamper(tmp.name)
            elif AVAILABLE["patch_model"]:
                t_res = infer_patch_tamper(tmp.name)
            else:
                t_res = {"prob_tampered":0.0, "tamper_label":"Unavailable", "confidence":0.0}

            img = Image.open(tmp.name).convert("RGB")
            c1,c2 = st.columns([2,1])
            with c1:
                st.image(img, use_column_width=True, caption=f"Scanner: {s_label} ({s_conf:.2f}%)")
            with c2:
                st.metric("Tamper", t_res.get("tamper_label","Unknown"))
                st.write(f"Probability: {t_res.get('prob_tampered',0.0):.3f}")
                if "hits" in t_res:
                    st.write("Hits:", t_res["hits"])
                st.write(f"Confidence: {t_res.get('confidence',0.0):.1f}%")
        finally:
            try: os.remove(tmp.name)
            except: pass

elif page == "About":
    st.title("About AI TraceFinder")
    st.markdown("""
    **AI TraceFinder** — Dashboard for digital image forensics.

    This application explores two fundamental challenges in digital image forensics:
    
    1.  **Source Scanner Identification:**
        * **What:** Identifies the specific flatbed scanner model used to create an image.
        * **How:** Uses a hybrid model combining a CNN (Convolutional Neural Network) with 27 handcrafted features (like LBP, FFT, and wavelet residuals) to capture the unique 'fingerprint' left by a scanner's sensor.

    2.  **Image Tamper Detection:**
        * **What:** Determines if an image has been digitally altered or manipulated.
        * **How:** Employs an 18-feature SVM for image-level analysis, with a more granular 22-feature patch-based SVM as a fallback to locate potential inconsistencies.

    **Purpose & Context:**
    This tool is designed for educational and research purposes to demonstrate core concepts in media forensics. In fields like criminal investigation, journalism, and intellectual property, verifying a document's authenticity and origin (its **provenance**) is critical. This dashboard provides a hands-on interface to see how subtle, often invisible, traces left by hardware can be used to answer these questions.

    ---
   
    """)
    st.write(f"Files loaded from: `{str(DATA_DIR)}`")
    st.markdown("---")
    st.markdown("*Project by Parth*")

# ---------------------------
# End
# -------------------------