# processing.py
import os, csv
from pathlib import Path

DATA_ROOT = os.getcwd()  # Must run from Tampered images
MANIFEST_CSV = os.path.join(DATA_ROOT,"tamper_manifest.csv")

def build_tamper_manifest(root=DATA_ROOT, out_csv=MANIFEST_CSV):
    rows = [["path","label","domain","tamper_type","page_id"]]
    
    # Original images
    ORIG = os.path.join(root,"Original")
    for dirpath, _, files in os.walk(ORIG):
        for f in files:
            if f.lower().endswith((".tif",".tiff",".png")):
                rows.append([os.path.join(dirpath,f),0,"tamper",Path(dirpath).name,Path(f).stem])
    
    # Tampered images
    TAMP = os.path.join(root,"Tampered")
    for dirpath, _, files in os.walk(TAMP):
        for f in files:
            if f.lower().endswith((".tif",".tiff",".png")):
                rows.append([os.path.join(dirpath,f),1,"tamper",Path(dirpath).name,Path(f).stem])
    
    # Binary masks
    BM = os.path.join(root,"Binary masks")
    for dirpath, _, files in os.walk(BM):
        for f in files:
            if f.lower().endswith((".tif",".tiff",".png")):
                rows.append([os.path.join(dirpath,f),1,"tamper",Path(dirpath).name,Path(f).stem])
    
    os.makedirs(os.path.dirname(out_csv),exist_ok=True)
    with open(out_csv,"w",newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Tamper manifest created at: {out_csv}")

if __name__=="__main__":
    build_tamper_manifest()
