# testing.py
import os, pickle, numpy as np
from utils import load_to_residual, extract_patches, make_feat_vector

MODEL_DIR = os.path.join(os.getcwd(),"tamper_models")
TOPK_FRAC = 0.3
MAX_PATCHES = 16

def load_patch_model():
    with open(os.path.join(MODEL_DIR,"patch_scaler.pkl"),"rb") as f: scaler=pickle.load(f)
    with open(os.path.join(MODEL_DIR,"patch_svm_sig_calibrated.pkl"),"rb") as f: clf=pickle.load(f)
    return scaler, clf

def predict_image_tamper(img_path):
    scaler, clf = load_patch_model()
    res = load_to_residual(img_path)
    patches,_ = extract_patches(res,limit=MAX_PATCHES)
    feats = [make_feat_vector(p) for p in patches]
    if not feats: return {"tamper_score":0.0,"is_tampered":False}
    X = np.array(feats,dtype=np.float32)
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)[:,1]
    k = max(1,int(len(probs)*TOPK_FRAC))
    score = float(np.mean(np.sort(probs)[-k:]))
    return {"tamper_score":score,"is_tampered":bool(score>=0.5)}
