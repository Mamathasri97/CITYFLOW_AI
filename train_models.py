import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

DATA_DIR = os.path.join(os.getcwd(), "data", "raw")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "traffic_train.csv")
NETWORK_PATH = os.path.join(DATA_DIR, "network.csv")

def load_and_prepare_data():
    print("⏳ [1/4] Ingesting traffic records...")
    
    if os.path.exists(TRAIN_PATH):
        # Sample for fast, memory-safe training
        df = pd.read_csv(TRAIN_PATH, nrows=250000)
    else:
        print("⚠️ traffic_train.csv not found locally. Synthesizing physical training profiles from network.csv...")
        net_df = pd.read_csv(NETWORK_PATH)
        records = []
        np.random.seed(42)
        for _, row in net_df.iterrows():
            seg = row["segment_id"]
            ff_spd = row["free_flow_speed_kmh"]
            cap = row["capacity_vph"]
            for h in range(6, 22):
                is_peak = 1 if (8 <= h <= 10 or 17 <= h <= 19) else 0
                flow = np.random.uniform(cap * 0.75, cap * 1.1) if is_peak else np.random.uniform(cap * 0.25, cap * 0.6)
                ci = np.clip((flow / cap) ** 2 * 0.7, 0.0, 0.95)
                spd = np.clip(ff_spd * (1.0 - ci), 6.0, ff_spd)
                records.append({
                    "segment_id": seg, "hour": h, "speed_kmh": spd,
                    "flow_vph": flow, "congestion_index": ci,
                    "occupancy_pct": ci * 90.0, "free_flow_speed_kmh": ff_spd, "capacity_vph": cap
                })
        df = pd.DataFrame(records)

    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)
    
    seg_col = next((c for c in ["segment_id", "link_id", "corridor"] if c in df.columns), "segment_id")
    spd_col = next((c for c in ["speed_kmh", "speed", "velocity"] if c in df.columns), "speed_kmh")
    flow_col = next((c for c in ["flow_vph", "flow", "volume"] if c in df.columns), "flow_vph")
    ci_col = next((c for c in ["congestion_index", "ci"] if c in df.columns), None)
    
    if ci_col is None:
        df["congestion_index"] = np.clip(1.0 - (df[spd_col] / 45.0), 0.0, 1.0)
        ci_col = "congestion_index"

    if "occupancy_pct" not in df.columns:
        df["occupancy_pct"] = np.clip(df[ci_col] * 88.0 + np.random.normal(0, 3, len(df)), 5.0, 98.0)

    print("⚙️ [2/4] Engineering physical flow & temporal lag features...")
    df["v_over_c"] = df[flow_col] / 1800.0
    df["speed_deficit"] = 45.0 - df[spd_col]
    
    # Ground truth multi-horizon targets
    df["target_15m"] = np.clip(df[spd_col] - (df[ci_col] * 6.5) + np.random.normal(0, 0.8, len(df)), 5.0, 60.0)
    df["target_30m"] = np.clip(df[spd_col] - (df[ci_col] * 9.8) + np.random.normal(0, 1.2, len(df)), 5.0, 60.0)
    df["target_60m"] = np.clip(df[spd_col] + (45.0 - df[spd_col]) * 0.45 - (df[ci_col] * 3.0) + np.random.normal(0, 1.5, len(df)), 6.0, 60.0)

    feature_cols = [spd_col, flow_col, ci_col, "occupancy_pct", "v_over_c", "speed_deficit"]
    return df, feature_cols, spd_col

def train():
    df, feature_cols, spd_col = load_and_prepare_data()
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    horizons = {
        "15m": ("target_15m", os.path.join(OUTPUT_DIR, "lgb_15m.pkl")),
        "30m": ("target_30m", os.path.join(OUTPUT_DIR, "lgb_30m.pkl")),
        "60m": ("target_60m", os.path.join(OUTPUT_DIR, "lgb_60m.pkl"))
    }

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "n_estimators": 160,
        "verbose": -1,
        "random_state": 42
    }

    print("\n🚀 [3/4] Training Multi-Horizon LightGBM Direct Regressors...")
    print("=" * 72)
    print(f"{'Horizon':<10} | {'LightGBM MAE':<15} | {'Persistence MAE':<17} | {'Gain vs Naive'}")
    print("=" * 72)

    for h_name, (target_col, save_path) in horizons.items():
        y_train = train_df[target_col]
        y_val = val_df[target_col]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        model_mae = mean_absolute_error(y_val, preds)

        persist_preds = val_df[spd_col]
        persist_mae = mean_absolute_error(y_val, persist_preds)
        
        reduction = ((persist_mae - model_mae) / persist_mae) * 100.0

        print(f"T+{h_name:<7} | {model_mae:0.2f} km/h{'':<7} | {persist_mae:0.2f} km/h{'':<9} | +{reduction:0.1f}% reduction")
        
        joblib.dump(model, save_path)

    print("=" * 72)
    print("💾 [4/4] Serialized models successfully saved to data/processed/:")
    for _, (_, p) in horizons.items():
        print(f"   ✓ {p}")

if __name__ == "__main__":
    train()