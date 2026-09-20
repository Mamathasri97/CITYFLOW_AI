import os
import pandas as pd
import json

raw_dir = "data/raw"

print("=" * 60)
print("DATASET AUDIT & COLUMN PROFILING")
print("=" * 60)

if not os.path.exists(raw_dir):
    print(f"Directory {raw_dir} not found. Ensure you are running from the project root.")
else:
    for fname in sorted(os.listdir(raw_dir)):
        fpath = os.path.join(raw_dir, fname)
        if fname.endswith(".csv"):
            try:
                df = pd.read_csv(fpath, nrows=5)
                full_df = pd.read_csv(fpath)
                print(f"\n📂 File: {fname}")
                print(f"   Shape: {full_df.shape[0]} rows, {full_df.shape[1]} cols")
                print(f"   Columns: {list(df.columns)}")
                print("   Sample:")
                print(df.head(2).to_string(index=False))
            except Exception as e:
                print(f"   Error reading {fname}: {e}")
        elif fname.endswith(".json"):
            print(f"\n📄 Manifest: {fname}")
            try:
                with open(fpath) as f:
                    manifest = json.load(f)
                    print("   Keys:", list(manifest.keys()))
            except Exception as e:
                print(f"   Error reading {fname}: {e}")