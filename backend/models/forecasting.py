import os
import joblib
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any

MODEL_DIR = "data/processed"

class MultiHorizonForecaster:
    def __init__(self):
        self.models = {}
        self.horizons = ["15m", "30m", "60m"]
        os.makedirs(MODEL_DIR, exist_ok=True)

    def train_quick_baseline(self, data_dir: str = "data/raw"):
        print("Training multi-horizon forecast models on organizer targets...")
        targets_df = pd.read_csv(os.path.join(data_dir, "forecast_targets_train.csv"), nrows=100000)
        traffic_df = pd.read_csv(os.path.join(data_dir, "traffic_train.csv"), nrows=100000)

        merged = pd.merge(
            traffic_df[["timestamp", "segment_id", "speed_kmh", "flow_vph", "congestion_index", "occupancy_pct"]],
            targets_df[["timestamp", "segment_id", "target_speed_15m", "target_speed_30m", "target_speed_60m"]],
            on=["timestamp", "segment_id"]
        ).dropna()

        X = merged[["speed_kmh", "flow_vph", "congestion_index", "occupancy_pct"]]

        for h in self.horizons:
            target_col = f"target_speed_{h}"
            y = merged[target_col]
            reg = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.08, random_state=42, verbose=-1)
            reg.fit(X, y)
            self.models[h] = reg
            joblib.dump(reg, os.path.join(MODEL_DIR, f"lgb_{h}.pkl"))
            print(f" Horizon +{h} model saved.")

    def load_models(self):
        for h in self.horizons:
            path = os.path.join(MODEL_DIR, f"lgb_{h}.pkl")
            if os.path.exists(path):
                self.models[h] = joblib.load(path)

    def predict(self, speed_kmh: float, flow_vph: float, congestion_index: float, occupancy_pct: float) -> Dict[str, float]:
        feats = pd.DataFrame([[speed_kmh, flow_vph, congestion_index, occupancy_pct]],
                             columns=["speed_kmh", "flow_vph", "congestion_index", "occupancy_pct"])
        preds = {}
        for h in self.horizons:
            if h in self.models:
                preds[h] = round(float(self.models[h].predict(feats)[0]), 2)
            else:
                preds[h] = round(speed_kmh, 2)
        return preds

if __name__ == "__main__":
    forecaster = MultiHorizonForecaster()
    forecaster.train_quick_baseline()