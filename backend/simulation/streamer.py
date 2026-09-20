import os
import time
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

def run_realtime_stream(validation_csv="data/raw/traffic_validation.csv", delay_seconds=2.0):
    if not os.path.exists(validation_csv):
        print(f"Error: {validation_csv} not found.")
        return

    print("Loading traffic validation stream...")
    # Load validation stream sorted by timestamp
    df = pd.read_csv(validation_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Jump straight to morning rush hour (8:00 AM onwards) where congestion & incidents occur
    df = df[df["timestamp"].dt.hour >= 8]
    timestamps = sorted(df["timestamp"].unique())

    if not timestamps:
        print("No records found for the specified time window.")
        return

    print(f"Loaded {len(timestamps)} time steps starting at morning peak {timestamps[0]}.")
    print(f"Commencing real-time playback (1 tick = {delay_seconds}s)...")

    for ts in timestamps:
        ts_str = str(ts)
        slice_df = df[df["timestamp"] == ts]
        print(f"\n⏱️ [SIMULATED CLOCK]: {ts_str} | Broadcasting {len(slice_df)} sensor updates...")

        # 1. Push link states to graph
        for _, row in slice_df.iterrows():
            seg_id = str(row["segment_id"])
            speed = float(row["speed_kmh"])
            ci = float(row["congestion_index"])
            occ = float(row["occupancy_pct"])

            # Update link state in the backend graph
            try:
                requests.post(
                    f"{API_URL}/api/network/update",
                    json={"segment_id": seg_id, "speed_kmh": speed, "congestion_index": ci},
                    timeout=1.0
                )
            except Exception:
                pass

            # 2. Evaluate high-congestion links for real-time incidents
            if ci >= 0.50:
                try:
                    inc_res = requests.post(
                        f"{API_URL}/api/incidents/evaluate",
                        json={"segment_id": seg_id, "current_speed_kmh": speed, "current_occupancy_pct": occ},
                        timeout=1.0
                    ).json()

                    if inc_res.get("incident_detected"):
                        print(f"   🚨 INCIDENT ALERT on {seg_id}! Z-Score: {inc_res['z_score']}σ | Conf: {inc_res['confidence']*100:.0f}%")
                except Exception:
                    pass

        time.sleep(delay_seconds)

if __name__ == "__main__":
    run_realtime_stream()