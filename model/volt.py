import glob
import numpy as np
from tqdm import tqdm

# Update this path to your actual data
files = glob.glob("hackathon_dataset_npz_final/train/*.npz")

max_val = -float('inf')

print(f" Scanning {len(files)} files for Peak Voltage...")

for f in tqdm(files):
    try:
        with np.load(f) as data:
            tgt = data['potential']
            curr_max = tgt.max()
            if curr_max > max_val:
                max_val = curr_max
    except:
        pass

print(f"\n MAXIMUM POTENTIAL IN DATASET: {max_val:.2f}")
print(f" Safe Scaling Factor to use: {max_val * 1.1:.1f}")