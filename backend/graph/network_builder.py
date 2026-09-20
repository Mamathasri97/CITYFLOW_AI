import os
import pandas as pd
import networkx as nx
from typing import Dict, Any, Tuple

class RoadNetwork:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.graph = nx.DiGraph()
        self.nodes_df = None
        self.network_df = None
        self.segment_to_nodes: Dict[str, Tuple[str, str]] = {}
        self.turn_restrictions = set()
        self.load_network()

    def load_network(self):
        nodes_path = os.path.join(self.data_dir, "nodes.csv")
        network_path = os.path.join(self.data_dir, "network.csv")
        turns_path = os.path.join(self.data_dir, "turn_restrictions.csv")

        self.nodes_df = pd.read_csv(nodes_path)
        for _, row in self.nodes_df.iterrows():
            self.graph.add_node(
                str(row["node_id"]),
                x=float(row["x"]),
                y=float(row["y"]),
                lat=float(row["lat"]),
                lon=float(row["lon"])
            )

        self.network_df = pd.read_csv(network_path)
        for _, row in self.network_df.iterrows():
            u = str(row["source_node"])
            v = str(row["target_node"])
            seg_id = str(row["segment_id"])
            length_km = float(row["length_km"])
            ff_speed = float(row["free_flow_speed_kmh"])
            capacity = float(row["capacity_vph"])
            lanes = int(row["lanes"])
            bottleneck = int(row["structural_bottleneck"])

            free_flow_time_min = (length_km / max(ff_speed, 1.0)) * 60.0

            self.graph.add_edge(
                u, v,
                segment_id=seg_id,
                length_km=length_km,
                free_flow_speed_kmh=ff_speed,
                current_speed_kmh=ff_speed,
                capacity_vph=capacity,
                lanes=lanes,
                free_flow_time_min=free_flow_time_min,
                current_travel_time_min=free_flow_time_min,
                congestion_index=0.0,
                structural_bottleneck=bottleneck,
                has_incident=False
            )
            self.segment_to_nodes[seg_id] = (u, v)

        if os.path.exists(turns_path):
            turns_df = pd.read_csv(turns_path)
            for _, row in turns_df.iterrows():
                if row.get("restriction") == "no_turn":
                    self.turn_restrictions.add((str(row["from_segment"]), str(row["to_segment"])))

    def update_edge_state(self, segment_id: str, speed_kmh: float, congestion_index: float):
        if segment_id in self.segment_to_nodes:
            u, v = self.segment_to_nodes[segment_id]
            length = self.graph[u][v]["length_km"]
            speed = max(speed_kmh, 1.0)
            travel_time = (length / speed) * 60.0
            self.graph[u][v]["current_speed_kmh"] = speed
            self.graph[u][v]["current_travel_time_min"] = travel_time
            self.graph[u][v]["congestion_index"] = max(0.0, min(1.0, congestion_index))

    def get_edge_attributes(self, segment_id: str) -> Dict[str, Any]:
        if segment_id in self.segment_to_nodes:
            u, v = self.segment_to_nodes[segment_id]
            return dict(self.graph[u][v])
        return {}