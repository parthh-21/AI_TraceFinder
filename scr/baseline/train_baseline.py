# train_baseline.py
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==========================
# Paths
# ==========================
BASE = r"C:\Users\parth\Desktop\AI_track_finder"
OFF_CSV = os.path.join(BASE, "processed_data", "Official", "metadata_features.csv")
FLAT_CSV = os.path.join(BASE, "processed_data", "Flatfield", "metadata_features.csv")
OUT_MODELS = os.path.join(BASE, "models")
os.makedirs(OUT_MODELS, exist_ok=True)

# ==========================
# Load CSVs
# ==========================
dfs = []
if os.path.exists(OFF_CSV):
    dfs.append(pd.read_csv(OFF_CSV))
if os.path.exists(FLAT_CSV):
    dfs.append(pd.read_csv(FLAT_CSV))

if not dfs:
    raise SystemExit("❌ No CSV files found. Run preprocessing first!")

df = pd.concat(dfs, ignore_index=True)

# ==========================
# Prepare features
# ==========================
X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
y = df["class_label"]

# Fill missing values
X = X.fillna(0)

# ==========================
# Scale features
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# Train Random Forest
# ==========================
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)
joblib.dump(rf, os.path.join(OUT_MODELS, "random_forest.pkl"))
print("✅ Random Forest model trained and saved!")

# ==========================
# Train SVM
# ==========================
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
svm.fit(X_scaled, y)
joblib.dump(svm, os.path.join(OUT_MODELS, "svm.pkl"))
print("✅ SVM model trained and saved!")

# ==========================
# Save scaler
# ==========================
joblib.dump(scaler, os.path.join(OUT_MODELS, "scaler.pkl"))
print("✅ Scaler saved!")
print("✅ All models and scaler ready for prediction and evaluation.")
