import networkx as nx
from typing import Dict, Any, List, Optional

class CapacitySafeRoutingEngine:
    """
    Capacity-constrained dynamic routing engine.
    Calculates bypass detours around blocked or severely congested corridors
    while enforcing the Secondary Congestion Guard (rejecting paths with CI >= 0.45).
    """
    def __init__(self, road_network):
        self.network = road_network
        self.graph = road_network.graph

    def calculate_diversion(self, blocked_segment_id: str, max_headroom_ci: float = 0.45) -> Dict[str, Any]:
        # 1. Locate blocked edge in the graph
        edge_found = None
        for u, v, data in self.graph.edges(data=True):
            if data.get("segment_id") == blocked_segment_id:
                edge_found = (u, v, data)
                break

        if not edge_found:
            return {
                "blocked_segment_id": blocked_segment_id,
                "detour_approved": False,
                "guard_rationale": f"Segment {blocked_segment_id} not found in road network graph."
            }

        u_orig, v_orig, orig_data = edge_found

        # 2. Build temporary routing graph with the blocked link removed
        temp_graph = self.graph.copy()
        if temp_graph.has_edge(u_orig, v_orig):
            temp_graph.remove_edge(u_orig, v_orig)

        # 3. Find candidate bypass path minimizing dynamic travel impedance
        try:
            path = nx.shortest_path(temp_graph, source=u_orig, target=v_orig, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {
                "blocked_segment_id": blocked_segment_id,
                "detour_approved": False,
                "guard_rationale": "No topological alternative path exists connecting origin to destination."
            }

        # 4. Secondary Congestion Guard: Audit each link along the candidate detour
        detour_segments = []
        max_ci = 0.0
        total_time_s = 0.0

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            edge_attrs = self.graph[src][dst]
            seg_id = edge_attrs.get("segment_id", f"{src}->{dst}")
            ci = edge_attrs.get("congestion_index", 0.0)
            length_m = edge_attrs.get("length_m", 500.0)
            speed_kmh = edge_attrs.get("current_speed_kmh", edge_attrs.get("free_flow_speed_kmh", 45.0))
            
            speed_mps = max(1.0, speed_kmh * (1000.0 / 3600.0))
            total_time_s += (length_m / speed_mps)

            if ci > max_ci:
                max_ci = ci
            detour_segments.append(seg_id)

        # Enforce threshold
        if max_ci >= max_headroom_ci:
            return {
                "blocked_segment_id": blocked_segment_id,
                "detour_approved": False,
                "max_corridor_ci": round(max_ci, 2),
                "threshold_ci": max_headroom_ci,
                "detour_path_nodes": path,
                "guard_rationale": (
                    f"Detour rejected: Candidate corridor contains link near saturation "
                    f"(Max CI: {max_ci:.2f} >= threshold {max_headroom_ci}). Diverting traffic would induce secondary gridlock."
                )
            }

        return {
            "blocked_segment_id": blocked_segment_id,
            "detour_approved": True,
            "detour_path_nodes": path,
            "detour_segments": detour_segments,
            "max_corridor_ci": round(max_ci, 2),
            "estimated_travel_time_min": round(total_time_s / 60.0, 1),
            "guard_rationale": f"Capacity headroom verified. Max bypass CI ({max_ci:.2f}) < {max_headroom_ci}."
        }

# Alias for backwards compatibility
RoutingEngine = CapacitySafeRoutingEngine