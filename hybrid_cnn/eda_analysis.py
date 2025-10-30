# ============================================================
#  eda_analysis.py — Updated Version (Parth’s paths & safe keys)
# ============================================================

import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: seaborn for nicer plots
try:
    import seaborn as sns
    sns_available = True
except ImportError:
    print("⚠️ Seaborn not installed. Plots will use matplotlib only.")
    sns_available = False

# ---------- PATHS ----------
BASE_DIR = r"C:\Users\parth\Desktop\AI_track_finder\dataset"
FEATURE_PATH = os.path.join(BASE_DIR, "enhanced_features.pkl")

# ---------- LOAD FEATURES ----------
with open(FEATURE_PATH, "rb") as f:
    data = pickle.load(f)

# Safe key access & convert to NumPy arrays
if "features" in data and "labels" in data:
    X = np.array(data["features"], dtype=np.float32)
    y = np.array(data["labels"])
elif "X" in data and "y" in data:
    X = np.array(data["X"], dtype=np.float32)
    y = np.array(data["y"])
else:
    raise KeyError(
        "❌ Could not find features and labels in the pickle file. "
        "Available keys: " + ", ".join(data.keys())
    )

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Unique scanner labels: {len(np.unique(y))}")

# ---------- CREATE DATAFRAME ----------
df = pd.DataFrame(X, columns=[f"F{i+1}" for i in range(X.shape[1])])
df["Scanner"] = y

# ---------- SCANNER DISTRIBUTION ----------
plt.figure(figsize=(10,5))
if sns_available:
    sns.countplot(x="Scanner", data=df, order=df["Scanner"].value_counts().index)
else:
    counts = df["Scanner"].value_counts()
    plt.bar(counts.index, counts.values)
plt.xticks(rotation=45)
plt.title("📊 Image Count per Scanner Type")
plt.tight_layout()
plt.show()

# ---------- FEATURE STATS ----------
desc = df.describe().T
print("\n📈 Feature Summary Statistics (first 5 features):\n", desc.head())

# ---------- CORRELATION HEATMAP ----------
plt.figure(figsize=(12,10))
corr_matrix = df.iloc[:,:15].corr()
if sns_available:
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
else:
    plt.imshow(corr_matrix, cmap="coolwarm", interpolation='nearest')
    plt.colorbar()
plt.title("🔍 Feature Correlation (first 15 features)")
plt.show()

# ---------- OPTIONAL: FEATURE DISTRIBUTIONS ----------
plt.figure(figsize=(15,10))
for i in range(min(6, X.shape[1])):  # show first 6 features
    plt.subplot(2,3,i+1)
    plt.hist(df[f"F{i+1}"], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Feature F{i+1} Distribution")
plt.tight_layout()
plt.show()
