import os
import numpy as np

DATA_DIR = r"C:\Users\parth\Desktop\AI_track_finder\processed_data\Official"

# Only pick first scanner folder
scanner_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
first_scanner = scanner_folders[0]

files = os.listdir(os.path.join(DATA_DIR, first_scanner))
npy_files = [f for f in files if f.endswith(".npy")]

print(f"Scanner: {first_scanner}, .npy files: {len(npy_files)}")

# Load just 5 files to check shapes
for f in npy_files[:5]:
    data = np.load(os.path.join(DATA_DIR, first_scanner, f))
    print(f"{f} shape: {data.shape}")
