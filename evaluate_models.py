import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from backend.models.forecasting import MultiHorizonForecaster
from backend.models.incident_detector import IncidentDetector
from backend.graph.network_builder import RoadNetwork

print("=" * 65)
print("CITYFLOW AI — CHECKPOINT 3 VALIDATION & ROBUSTNESS AUDIT")
print("=" * 65)

data_dir = "data/raw"

# 1. Initialize Network and Forecaster
net = RoadNetwork(data_dir=data_dir)
forecaster = MultiHorizonForecaster()
forecaster.load_models()

# 2. Evaluate Traffic Forecasting (Vectorized Batch)
print("\n--- 1. FORECASTING ACCURACY (Unseen Validation Set) ---")
traffic_val = pd.read_csv(os.path.join(data_dir, "traffic_validation.csv"), nrows=50000)
targets_val = pd.read_csv(os.path.join(data_dir, "forecast_targets_validation.csv"), nrows=50000)

merged = pd.merge(
    traffic_val[["timestamp", "segment_id", "speed_kmh", "flow_vph", "congestion_index", "occupancy_pct"]],
    targets_val[["timestamp", "segment_id", "target_speed_15m", "target_speed_30m", "target_speed_60m"]],
    on=["timestamp", "segment_id"]
).dropna()

X_val = merged[["speed_kmh", "flow_vph", "congestion_index", "occupancy_pct"]]

results = {}
for h in ["15m", "30m", "60m"]:
    actuals = merged[f"target_speed_{h}"]
    # Fast vectorized batch prediction
    preds = forecaster.models[h].predict(X_val)
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    results[h] = {"mae": mae, "rmse": rmse}
    print(f"Horizon +{h:>3} | MAE: {mae:.2f} km/h | RMSE: {rmse:.2f} km/h")

# 3. Robustness Under Synthetic Sensor Noise
print("\n--- 2. ROBUSTNESS UNDER NOISY / PERTURBED SENSOR DATA ---")
X_val_noisy = X_val.copy()
noise = np.random.normal(0, 5.0, len(X_val_noisy))
X_val_noisy["speed_kmh"] = np.clip(X_val_noisy["speed_kmh"] + noise, 2.0, 90.0)

noisy_actuals = merged["target_speed_15m"]
noisy_preds = forecaster.models["15m"].predict(X_val_noisy)
noisy_mae = mean_absolute_error(noisy_actuals, noisy_preds)
print(f"Horizon +15m under ±5 km/h Sensor Noise -> MAE: {noisy_mae:.2f} km/h (Degradation: {noisy_mae - results['15m']['mae']:+.2f} km/h)")

# 4. Incident Detection & False Alarm Control
print("\n--- 3. INCIDENT DETECTION & FALSE ALARM CONTROL ---")
detector = IncidentDetector(net)
train_sample = pd.read_csv(os.path.join(data_dir, "traffic_train.csv"), nrows=50000)
detector.fit_baseline(train_sample)

incidents_val = pd.read_csv(os.path.join(data_dir, "incidents_validation.csv"))
ground_truth_segments = set(incidents_val["segment_id"].dropna().astype(str))

sample_eval = traffic_val.sample(n=1000, random_state=42)
y_true = [1 if str(row["segment_id"]) in ground_truth_segments and row["congestion_index"] >= 0.40 else 0 for _, row in sample_eval.iterrows()]
y_pred = []

for _, row in sample_eval.iterrows():
    evaluation = detector.evaluate_segment(str(row["segment_id"]), float(row["speed_kmh"]), float(row["occupancy_pct"]))
    y_pred.append(1 if evaluation["incident_detected"] else 0)

prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
false_alarms = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
far = (false_alarms / len(y_true)) * 100

print(f"Incident Detection Precision : {prec * 100:.1f}%")
print(f"Incident Detection Recall    : {rec * 100:.1f}%")
print(f"Incident Detection F1-Score  : {f1:.3f}")
print(f"Corridor False Alarm Rate    : {far:.2f}%")

print("\n" + "=" * 65)
print("AUDIT COMPLETE: All metrics ready for README.md Checkpoint 3.")
print("=" * 65)