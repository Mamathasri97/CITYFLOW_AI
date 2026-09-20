import os
import copy
import pandas as pd
import networkx as nx
from typing import Dict, Any

class WhatIfSimulator:
    def __init__(self, base_network, data_dir: str = "data/raw"):
        self.base_network = base_network
        self.data_dir = data_dir
        self.candidates_df = pd.read_csv(os.path.join(data_dir, "planning_candidates.csv"))

    def evaluate_candidate(self, candidate_id: str) -> Dict[str, Any]:
        match = self.candidates_df[self.candidates_df["candidate_id"] == candidate_id]
        if match.empty:
            return {"error": f"Candidate {candidate_id} not found."}

        row = match.iloc[0]
        target_segment = str(row["target_segment"])
        added_capacity = float(row["capacity_delta_vph"])
        cost_idx = float(row["cost_index"])

        sim_net = copy.deepcopy(self.base_network.graph)

        u, v = self.base_network.segment_to_nodes.get(target_segment, (None, None))
        if not u or not v or not sim_net.has_edge(u, v):
            return {"error": f"Target segment {target_segment} not found in graph."}

        orig_cap = sim_net[u][v]["capacity_vph"]
        new_cap = orig_cap + added_capacity
        sim_net[u][v]["capacity_vph"] = new_cap

        orig_time = sim_net[u][v]["current_travel_time_min"]
        improved_time = max(sim_net[u][v]["free_flow_time_min"], orig_time * (orig_cap / new_cap))
        sim_net[u][v]["current_travel_time_min"] = improved_time

        volume = orig_cap * 0.75
        delta_delay = (improved_time - orig_time) * volume

        return {
            "candidate_id": candidate_id,
            "target_segment": target_segment,
            "intervention_type": row["intervention_type"],
            "original_capacity_vph": orig_cap,
            "new_capacity_vph": new_cap,
            "delay_reduction_veh_min": round(abs(delta_delay), 2),
            "cost_index": cost_idx,
            "feasibility": row["feasibility_band"]
        }