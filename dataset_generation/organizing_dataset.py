import os, re, shutil, hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

SOURCE_FOLDERS = [
    "/home/s/sri007/my_project/microscopy/dataset_outputs_mod_batch",
    "/home/s/sri007/my_project/microscopy/dataset_outputs_mod",
    "/home/s/sri007/my_project/microscopy/dataset_outputs",
    "/home/s/sri007/my_project/microscopy/downloaded_data",
    "/home/s/sri007/my_project/microscopy/munich_data"

]

DEST_DIR = "hackathon_dataset_npz_final"
VAL_SPLIT_RATIO = 0.1
SEED = 42

def decode_npz_string(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
        else:
            x = x.flat[0].item()
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)

def safe_name(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120]

def short_hash(p):
    return hashlib.md5(str(p).encode()).hexdigest()[:8]

def consolidate_data():
    train_dir = os.path.join(DEST_DIR, "train")
    val_dir = os.path.join(DEST_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    material_groups = defaultdict(list)

    print(f" Scanning folders: {SOURCE_FOLDERS}")

    all_files = []
    for folder in SOURCE_FOLDERS:
        all_files.extend(list(Path(folder).rglob("*.npz")))

    print(f" Found {len(all_files)} .npz files. Inspecting metadata...")

    valid_files = 0
    for fpath in tqdm(all_files):
        try:
            with np.load(fpath, allow_pickle=True) as data:
                if "material" in data:
                    mat_name = decode_npz_string(data["material"])
                else:
                    mat_name = "Unknown"
                material_groups[mat_name].append(fpath)
                valid_files += 1
        except Exception as e:
            print(f"⚠️ Corrupt file {fpath}: {e}")

    print(f" Sorted {valid_files} files into {len(material_groups)} material groups.")

    rng = np.random.default_rng(SEED)

    copied_count = 0
    for mat, files in material_groups.items():
        files = list(files)
        rng.shuffle(files)

        split_idx = int(len(files) * (1 - VAL_SPLIT_RATIO))
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        mat_safe = safe_name(mat)

        def copy_batch(file_list, split_dir):
            nonlocal copied_count
            for f in file_list:
                new_name = f"{mat_safe}__{short_hash(f)}__{f.name}"
                dest_path = os.path.join(split_dir, new_name)
                shutil.copy2(f, dest_path)
                copied_count += 1

        copy_batch(train_files, train_dir)
        copy_batch(val_files, val_dir)

    print(f" Done! {copied_count} files consolidated in '{DEST_DIR}'.")

if __name__ == "__main__":
    consolidate_data()