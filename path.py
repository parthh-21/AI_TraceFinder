import os

# Project folder path
project_folder = r"C:\Users\parth\Desktop\AI_track_finder"

# Folders to ignore
ignore_folders = {'.venv', 'bekkar', '.git'}

# File extensions to ignore
ignore_extensions = {'.tif', '.jpg', '.png', '.pdf'}

# Minimum file size to include (in bytes)
min_size = 10 * 1024 * 1024  # 10 MB

# Walk through folder and subfolders
for root, dirs, files in os.walk(project_folder):
    # Skip ignored folders
    dirs[:] = [d for d in dirs if d not in ignore_folders]

    for file in files:
        if not any(file.lower().endswith(ext) for ext in ignore_extensions):
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) >= min_size:
                print(file_path)
