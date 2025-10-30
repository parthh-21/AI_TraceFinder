import os
import csv

# === BASE DIRECTORY (update only if you moved project) ===
BASE_DIR = r"C:\Users\parth\Desktop\AI_track_finder"

# === OUTPUT CSV FILE ===
output_csv = os.path.join(BASE_DIR, "all_files_paths.csv")

# === COLLECT FILE PATHS ===
all_files = []
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        rel_path = os.path.relpath(file_path, BASE_DIR)
        all_files.append([file, file_path, rel_path])

# === WRITE TO CSV ===
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File Name", "Full Path", "Relative Path"])
    writer.writerows(all_files)

print(f"✅ File listing completed! Found {len(all_files)} files.")
print(f"📁 Output saved to: {output_csv}")

# === Optional: Preview the first few entries ===
print("\n--- Sample entries ---")
for row in all_files[:10]:
    print(row)
