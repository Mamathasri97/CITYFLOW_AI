import numpy as np
import pandas as pd
from typing import Dict, List, Any

class IncidentDetector:
    def __init__(self, road_network):
        self.net = road_network
        self.history_stats: Dict[str, Dict[str, float]] = {}

    def fit_baseline(self, traffic_df: pd.DataFrame):
        grouped = traffic_df.groupby("segment_id")["speed_kmh"].agg(["mean", "std"]).reset_index()
        for _, row in grouped.iterrows():
            std_val = row["std"] if (pd.notnull(row["std"]) and row["std"] > 1.0) else 5.0
            self.history_stats[str(row["segment_id"])] = {
                "mean": float(row["mean"]),
                "std": float(std_val)
            }

    def evaluate_segment(self, segment_id: str, current_speed: float, current_occ: float) -> Dict[str, Any]:
        if segment_id not in self.history_stats:
            return {"incident_detected": False, "confidence": 0.0, "evidence": []}

        hist = self.history_stats[segment_id]
        mean_spd, std_spd = hist["mean"], hist["std"]
        z_score = (mean_spd - current_speed) / std_spd

        edge_data = self.net.get_edge_attributes(segment_id)
        ci = edge_data.get("congestion_index", 0.0)

        evidence = []
        is_incident = False

        if z_score >= 2.5 and ci >= 0.50:
            evidence.append(f"Z-Score speed anomaly of {z_score:.2f}σ below baseline")
            evidence.append(f"Severe link Congestion Index: {ci:.2f}")

            u, v = self.net.segment_to_nodes.get(segment_id, (None, None))
            if u and v:
                upstream_edges = [
                    d["segment_id"] for p, _, d in self.net.graph.in_edges(u, data=True)
                ]
                downstream_edges = [
                    d["segment_id"] for _, n, d in self.net.graph.out_edges(v, data=True)
                ]

                upstream_crowded = any(
                    self.net.get_edge_attributes(e).get("congestion_index", 0) > 0.40 for e in upstream_edges
                )
                downstream_free = any(
                    self.net.get_edge_attributes(e).get("congestion_index", 0) < 0.30 for e in downstream_edges
                )

                if upstream_crowded and downstream_free:
                    evidence.append("Spatial corroboration: upstream queue accumulation with downstream starvation confirmed")
                    is_incident = True
                elif z_score >= 3.5:
                    evidence.append("Extreme velocity suppression beyond regular corridor saturation")
                    is_incident = True

        confidence = round(min(0.95, 0.50 + (z_score * 0.10)), 2) if is_incident else 0.0

        return {
            "incident_detected": is_incident,
            "segment_id": segment_id,
            "z_score": round(float(z_score), 2),
            "confidence": confidence,
            "evidence": evidence
        }