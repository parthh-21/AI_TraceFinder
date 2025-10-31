# ==========================================================
# streamlit_app.py  — AI TraceFinder (Educator Dark UI)
# ==========================================================
# Author: Parth Gomase
# GitHub: https://github.com/parthh-21
# Purpose: Unified UI for Baseline (RF/SVM), CNN, Hybrid + Tamper checks
# ==========================================================

import os, sys, subprocess, tempfile, json, time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Optional imports guarded
try:
    import tensorflow as tf
except Exception:
    tf = None

# --------------------------
# Project Paths (user-specific)
# --------------------------
PROJECT_ROOT = Path(r"C:\Users\parth\Desktop\AI_track_finder")
SCR_DIR = PROJECT_ROOT / "scr"
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Tamper / hybrid artifacts location (from your structure)
TAMPER_MODELS_DIR = DATASET_DIR / "Tampered images" / "tamper_models"
HYBRID_MODEL_PATH = TAMPER_MODELS_DIR / "scanner_hybrid.keras"
HYBRID_LE = TAMPER_MODELS_DIR / "hybrid_label_encoder.pkl"
HYBRID_FEAT_SCALER = TAMPER_MODELS_DIR / "hybrid_feat_scaler.pkl"
FINGERPRINTS = TAMPER_MODELS_DIR / "scanner_fingerprints.pkl"
FP_KEYS = TAMPER_MODELS_DIR / "fp_keys.npy"

# Baseline
BASELINE_RF = MODELS_DIR / "random_forest.pkl"
BASELINE_SVM = MODELS_DIR / "svm.pkl"
BASELINE_SCALER = MODELS_DIR / "scaler.pkl"
BASELINE_CSV = PROCESSED_DIR / "Official" / "metadata_features.csv"

# CNN results (optional)
CNN_RESULTS_JSON = RESULTS_DIR / "cnn_classification_report.json"
CNN_CONF_IMG = RESULTS_DIR / "cnn_confusion_matrix_27dim.png"
CNN_MODEL_FOLDER = SCR_DIR / "cnn" / "models"

# Tamper artifacts
PATCH_SCALER = TAMPER_MODELS_DIR / "patch_scaler.pkl"
PATCH_CLF = TAMPER_MODELS_DIR / "patch_svm_sig_calibrated.pkl"
IMAGE_SCALER = TAMPER_MODELS_DIR / "image_scaler.pkl"
IMAGE_CLF = TAMPER_MODELS_DIR / "image_svm_sig.pkl"
IMAGE_THRESH = TAMPER_MODELS_DIR / "image_thresholds.json"
PATCH_THRESH = TAMPER_MODELS_DIR / "thresholds_patch.json"

# Make sure paths exist (not creating, only for checks)
def exists(p):
    return p is not None and Path(p).exists()

# --------------------------
# Streamlit page config & theme friendly layout
# --------------------------
st.set_page_config(page_title="AI TraceFinder", layout="wide",
                   initial_sidebar_state="expanded")
# Title area
st.markdown("<h1 style='text-align:left; color: white;'>TraceFinder - Forensic Scanner Identification</h1>",
            unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("## TraceFinder")
page = st.sidebar.radio("Navigation", [
    "Home",
    "Dataset Overview",
    "Feature Visualization",
    "Model Training & Evaluation",
    "Live Prediction",
    "About"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Select Model")
model_type = st.sidebar.selectbox("Model", ["Baseline (RF)", "Baseline (SVM)", "CNN (Hybrid)", "Hybrid (CNN+27D)"])

st.sidebar.caption("Models loaded status below 👇")

# Model status detection
status_msgs = {}
status_msgs["Baseline_RF"] = "Loaded" if exists(BASELINE_RF) else "Missing"
status_msgs["Baseline_SVM"] = "Loaded" if exists(BASELINE_SVM) else "Missing"
status_msgs["Baseline_Scaler"] = "Loaded" if exists(BASELINE_SCALER) else "Missing"
status_msgs["Hybrid_Model"] = "Loaded" if exists(HYBRID_MODEL_PATH) else "Missing"
status_msgs["Hybrid_Artifacts"] = "Loaded" if (exists(HYBRID_LE) and exists(HYBRID_FEAT_SCALER) and exists(FINGERPRINTS) and exists(FP_KEYS)) else "Missing"
status_msgs["PatchTamper"] = "Loaded" if (exists(PATCH_SCALER) and exists(PATCH_CLF)) else "Missing"
status_msgs["ImageTamper"] = "Loaded" if (exists(IMAGE_SCALER) and exists(IMAGE_CLF) and exists(IMAGE_THRESH)) else "Missing"

for k, v in status_msgs.items():
    st.sidebar.write(f"• {k.replace('_',' ')}: **{v}**")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**GitHub:** [parthh-21](https://github.com/parthh-21)")

# --------------------------
# Helper loaders (cached)
# --------------------------
@st.cache_resource
def load_joblib_model(path_str):
    try:
        return joblib.load(path_str)
    except Exception as e:
        return None

@st.cache_resource
def load_tf_model(path_str):
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(path_str, compile=False)
    except Exception:
        return None

@st.cache_resource
def load_pickle(path_str):
    try:
        import pickle
        with open(path_str, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None

@st.cache_resource
def load_json(path_str):
    try:
        with open(path_str, "r") as fh:
            return json.load(fh)
    except Exception:
        return None

# --------------------------
# Utility: compute scanner 27-D features (uses fingerprints if available)
# Minimal safe implementation: falls back to LBP+FFT if fingerprints missing
# --------------------------
def compute_residual_pywt(path, img_size=(256,256)):
    import cv2, pywt
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Cannot read image")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    return (img - den).astype("float32")

def corr2d(a,b):
    a = a.ravel().astype("float32"); b = b.ravel().astype("float32")
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d==0 else float((a @ b) / d)

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h,w = mag.shape
    cy,cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    feats=[]
    for i in range(K):
        m = (r>=bins[i]) & (r<bins[i+1])
        feats.append(float(mag[m].mean()) if m.any() else 0.0)
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    from skimage.feature import local_binary_pattern
    rng = float(np.ptp(img))
    g = (img - float(np.min(img))) / (rng + 1e-8) if rng>=1e-12 else np.zeros_like(img, dtype=np.float32)
    g8 = (g*255.0).astype("uint8")
    codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins+1), density=True)
    return hist.astype("float32")

def make_scanner_27d(res):
    # if fingerprint files present, use them; else fallback to LBP+FFT+zeros
    if exists(FINGERPRINTS) and exists(FP_KEYS):
        fps = load_pickle(str(FINGERPRINTS))
        keys = np.load(str(FP_KEYS), allow_pickle=True).tolist()
        v_corr = [corr2d(res, fps[k]) for k in keys]
    else:
        v_corr = [0.0]*11
    v_fft = fft_radial_energy(res, 6)
    v_lbp = lbp_hist_safe(res, 8, 1.0).tolist()  # returns 10
    feat = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
    return feat

# --------------------------
# Page: Home
# --------------------------
if page == "Home":
    st.markdown("### Purpose: Identify scanner used to scan a document by analyzing scanner-specific artifacts.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Total images", value="2200")
        st.metric("Scanner classes", value="11")
        st.metric("Example DPI levels", value="150, 300, 600")
        st.markdown("**Model status**")
        st.write(status_msgs)
    with col2:
        # educator image if present (use local screenshot if exists)
        local_example = PROJECT_ROOT / "Screenshot 2025-10-29 181608.png"
        if exists(local_example):
            st.image(str(local_example), caption="Parth Gomase | RSCOE", use_container_width=True)
        else:
            st.info("Home banner image not found. UI preview shown in sidebar.")

# --------------------------
# Page: Dataset Overview
# --------------------------
elif page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("Folders scanned in dataset directory:")
    for sub in ["Official", "flatfield", "Tampered images", "wiki dataset"]:
        p = DATASET_DIR / sub
        if p.exists():
            # count immediate children (folders only)
            n = len([x for x in p.iterdir() if x.is_dir()])
            st.write(f"• {sub} — {n} subfolders")
        else:
            st.warning(f"{sub} not found under {DATASET_DIR}")

    # Show Official metadata if exists
    if exists(BASELINE_CSV):
        try:
            df = pd.read_csv(str(BASELINE_CSV))
            st.markdown("**Official metadata (sample):**")
            st.dataframe(df.head(6))
        except Exception as e:
            st.warning(f"Could not read metadata CSV: {e}")

# --------------------------
# Page: Feature Visualization
# --------------------------
elif page == "Feature Visualization":
    st.header("Feature Visualization")
    if exists(BASELINE_CSV):
        df = pd.read_csv(str(BASELINE_CSV))
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            feature = st.selectbox("Choose numeric feature", numeric)
            st.write("Distribution for", feature)
            fig = None
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6,3))
                sns.histplot(df[feature].dropna(), ax=ax, kde=True)
                st.pyplot(fig)
            except Exception as e:
                st.write("Plotting error:", e)
        else:
            st.info("No numeric columns found in metadata CSV.")
    else:
        st.warning("Baseline metadata CSV not found; run preprocessing/CSV creation.")

# --------------------------
# Page: Model Training & Evaluation
# --------------------------
elif page == "Model Training & Evaluation":
    st.header("Train / Evaluate Models")

    st.markdown("### Baseline (Random Forest / SVM)")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train Baseline (run train_baseline.py)"):
            train_script = SCR_DIR / "baseline" / "train_baseline.py"
            if train_script.exists():
                with st.spinner("Training baseline..."):
                    result = subprocess.run(["python", str(train_script)], capture_output=True, text=True)
                    st.text(result.stdout[:10000])
                    if result.stderr:
                        st.text("Errors:\n" + result.stderr[:2000])
            else:
                st.error("train_baseline.py not found in scr/baseline.")

    with c2:
        if st.button("Evaluate Baseline (run evaluate_baseline.py)"):
            eval_script = SCR_DIR / "baseline" / "evaluate_baseline.py"
            if eval_script.exists():
                with st.spinner("Evaluating baseline..."):
                    result = subprocess.run(["python", str(eval_script)], capture_output=True, text=True)
                    st.text(result.stdout[:10000])
            else:
                st.error("evaluate_baseline.py not found.")

    st.markdown("---")
    st.markdown("### CNN/Hybrid")
    c3, c4 = st.columns(2)
    with c3:
        if st.button("Train CNN (run train_cnn.py)"):
            train_cnn = SCR_DIR / "cnn" / "train_cnn.py"
            if train_cnn.exists():
                with st.spinner("Training CNN..."):
                    r = subprocess.run(["python", str(train_cnn)], capture_output=True, text=True)
                    st.text(r.stdout[:10000])
            else:
                st.error("train_cnn.py not found under scr/cnn.")
    with c4:
        if st.button("Evaluate CNN (show results folder)"):
            if CNN_RESULTS_JSON.exists():
                st.success("CNN results found.")
                st.write("Open Results folder to view detailed outputs.")
            else:
                st.warning("No CNN results found in results/. Run training first.")

# --------------------------
# Page: Live Prediction
# --------------------------
elif page == "Live Prediction":
    st.header("Live Prediction — Upload an image")

    uploaded = st.file_uploader("Upload image (tif/png/jpg). For best results use TIFF scans", type=["tif", "tiff", "png", "jpg", "jpeg"])
    run_pred = st.button("Run")

    # Which model to apply: baseline RF/SVM, CNN hybrid, or hybrid inference
    chosen_model = st.selectbox("Choose Inference Engine", ["Baseline RF", "Baseline SVM", "CNN Hybrid (Keras)", "Hybrid 27-D + CNN"])

    if run_pred and uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read()); tmp.close()
        tmp_path = Path(tmp.name)
        try:
            # Baseline path
            if chosen_model == "Baseline RF":
                model = load_joblib_model(str(BASELINE_RF))
                scaler = load_joblib_model(str(BASELINE_SCALER))
                if model is None or scaler is None:
                    st.error("Baseline RF or scaler missing.")
                else:
                    import cv2
                    from skimage.filters import sobel
                    img = cv2.imread(str(tmp_path), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (512,512))
                    img = img.astype("float32")/255.0
                    pixels = img.flatten()
                    from scipy.stats import skew, kurtosis, entropy
                    feats = np.array([np.mean(pixels), np.std(pixels), skew(pixels), kurtosis(pixels),
                                      entropy(np.histogram(pixels, bins=256, range=(0,1))[0]+1e-6), np.mean(sobel(img)>0.1)]).reshape(1,-1)
                    Xs = scaler.transform(feats)
                    pred = model.predict(Xs)[0]
                    probs = model.predict_proba(Xs)[0]
                    st.image(Image.open(tmp_path), use_container_width=True, caption=f"Pred: {pred}")
                    st.write("Probabilities:", probs)

            # Baseline SVM
            elif chosen_model == "Baseline SVM":
                model = load_joblib_model(str(BASELINE_SVM))
                scaler = load_joblib_model(str(BASELINE_SCALER))
                if model is None or scaler is None:
                    st.error("Baseline SVM or scaler missing.")
                else:
                    import cv2
                    from skimage.filters import sobel
                    img = cv2.imread(str(tmp_path), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (512,512))
                    img = img.astype("float32")/255.0
                    pixels = img.flatten()
                    from scipy.stats import skew, kurtosis, entropy
                    feats = np.array([np.mean(pixels), np.std(pixels), skew(pixels), kurtosis(pixels),
                                      entropy(np.histogram(pixels, bins=256, range=(0,1))[0]+1e-6), np.mean(sobel(img)>0.1)]).reshape(1,-1)
                    Xs = scaler.transform(feats)
                    pred = model.predict(Xs)[0]
                    probs = model.predict_proba(Xs)[0]
                    st.image(Image.open(tmp_path), use_container_width=True, caption=f"Pred: {pred}")
                    st.write("Probabilities:", probs)

            # CNN Hybrid (Keras model only; predicts classes from residual image)
            elif chosen_model == "CNN Hybrid (Keras)":
                if not exists(HYBRID_MODEL_PATH):
                    st.error("Hybrid CNN model file missing (scanner_hybrid.keras)")
                else:
                    model = load_tf_model(str(HYBRID_MODEL_PATH))
                    if model is None:
                        st.error("Failed to load Keras model. Ensure TensorFlow is installed and model is accessible.")
                    else:
                        res = compute_residual_pywt(str(tmp_path))
                        x_img = np.expand_dims(res, axis=(0, -1))
                        # fallback: if model expects single input, try it; else model may be hybrid expecting 2 inputs
                        try:
                            prob = model.predict(x_img, verbose=0).ravel()
                            idx = int(np.argmax(prob))
                            st.image(Image.open(tmp_path), use_container_width=True)
                            st.success(f"CNN predicted class idx {idx} with conf {prob[idx]:.3f}")
                        except Exception:
                            st.error("Model predict failed: model may require feature input as well (use Hybrid option).")

            # Hybrid 27-D + CNN inference: compute 27-D features + call hybrid model
            elif chosen_model == "Hybrid 27-D + CNN":
                # check artifacts
                if not exists(HYBRID_MODEL_PATH):
                    st.error("Hybrid model (scanner_hybrid.keras) not found.")
                elif not (exists(HYBRID_FEAT_SCALER) and exists(HYBRID_LE)):
                    st.error("Hybrid scaler/label-encoder missing.")
                else:
                    model = load_tf_model(str(HYBRID_MODEL_PATH))
                    feat_scaler = load_pickle(str(HYBRID_FEAT_SCALER))
                    le = load_pickle(str(HYBRID_LE))
                    fps = load_pickle(str(FINGERPRINTS)) if exists(FINGERPRINTS) else None
                    keys = np.load(str(FP_KEYS), allow_pickle=True).tolist() if exists(FP_KEYS) else None

                    res = compute_residual_pywt(str(tmp_path))
                    ft = make_scanner_27d(res)
                    if hasattr(feat_scaler, "transform"):
                        ft_s = feat_scaler.transform(ft)
                    else:
                        ft_s = ft
                    # model takes [img, features] usually
                    x_img = np.expand_dims(res, axis=(0, -1))
                    try:
                        prob = model.predict([x_img, ft_s], verbose=0).ravel()
                        idx = int(np.argmax(prob))
                        label = le.classes_[idx] if (le is not None and hasattr(le, "classes_")) else str(idx)
                        st.image(Image.open(tmp_path), use_container_width=True)
                        st.success(f"Hybrid predicted: {label} ({prob[idx]*100:.2f}%)")
                    except Exception as e:
                        st.error(f"Hybrid prediction failed: {e}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

# --------------------------
# Page: About
# --------------------------
elif page == "About":
    st.header("About AI TraceFinder")
    st.markdown("""
    **AI TraceFinder** — Hybrid scanner identification & tamper detection system.
    - Developed by **Parth Gomase**
    - GitHub: [https://github.com/parthh-21](https://github.com/parthh-21)
    - Modules: Baseline (RF/SVM), Hybrid CNN (27-D + CNN), Tamper (patch + image)
    """)
    st.markdown("**Project paths used by this UI**")
    st.code(f"PROJECT_ROOT = {PROJECT_ROOT}\nMODELS_DIR = {MODELS_DIR}\nDATASET_DIR = {DATASET_DIR}")

# --------------------------
# Footer
# --------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Parth Gomase © 2025 | RSCOE | GitHub: parthh-21")
