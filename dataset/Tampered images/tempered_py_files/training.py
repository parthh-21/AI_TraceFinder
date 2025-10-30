# training.py
import os, pickle, csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from utils import load_to_residual, extract_patches, make_feat_vector

DATA_ROOT = os.getcwd()
MANIFEST_CSV = os.path.join(DATA_ROOT,"tamper_manifest.csv")
MODEL_DIR = os.path.join(DATA_ROOT,"tamper_models")
os.makedirs(MODEL_DIR,exist_ok=True)

def train_patch_svm():
    X, y = [], []
    with open(MANIFEST_CSV,newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["path"]
            label = int(row["label"])
            try:
                res = load_to_residual(path)
                patches,_ = extract_patches(res)
                for p in patches:
                    X.append(make_feat_vector(p))
                    y.append(label)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    X = np.array(X,dtype=np.float32)
    y = np.array(y,dtype=np.int64)
    print(f"Total patches: {X.shape[0]}, Feature size: {X.shape[1]}")
    
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    base = SVC(kernel="rbf",C=2.0,gamma="scale",class_weight="balanced")
    clf = CalibratedClassifierCV(base,cv=5,method="sigmoid")
    clf.fit(Xs,y)
    
    with open(os.path.join(MODEL_DIR,"patch_scaler.pkl"),"wb") as f: pickle.dump(scaler,f)
    with open(os.path.join(MODEL_DIR,"patch_svm_sig_calibrated.pkl"),"wb") as f: pickle.dump(clf,f)
    
    probs = clf.predict_proba(Xs)[:,1]
    auc = roc_auc_score(y,probs)
    print(f"Training AUC: {auc:.4f}")
    print(f"Model saved in: {MODEL_DIR}")

if __name__=="__main__":
    train_patch_svm()
