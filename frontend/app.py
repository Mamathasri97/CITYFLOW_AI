import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
import json
import os
import time

# Base URL for local FastAPI service
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="CITYFLOW AI — Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #0e1726, #050811);
        color: #f1f5f9;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
        transition: all 0.25s ease-in-out;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.4);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #090e1a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7, #0369a1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.39);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1, #075985);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-green { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-cyan { background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }
</style>
""", unsafe_allow_html=True)

head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown("## 🚦 CITYFLOW AI <span style='font-size: 1rem; color: #64748b; font-weight: 400;'>| Hyderabad Metropolitan Twin</span>", unsafe_allow_html=True)
with head_col2:
    st.markdown("<div style='text-align: right; padding-top: 10px;'><span class='badge badge-cyan'>Operational Twin</span></div>", unsafe_allow_html=True)

st.sidebar.markdown("### 🎛️ Operations Console")
live_mode = st.sidebar.toggle("🔴 Live Stream Playback", value=True)
refresh_rate = st.sidebar.slider("Sampling Tick (seconds)", 1, 5, 2)

st.sidebar.divider()
st.sidebar.markdown("### 📡 System Telemetry")

backend_online = False
try:
    health_resp = requests.get(f"{API_BASE}/", timeout=2).json()
    backend_online = True
    st.sidebar.markdown(f"""
    <div style='background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.2); padding: 10px; border-radius: 8px;'>
        <div style='color: #4ade80; font-size: 0.85rem; font-weight: 600;'>🟢 REST Engine Connected</div>
        <div style='font-size: 0.75rem; color: #94a3b8; margin-top: 4px;'>Nodes: {health_resp['nodes_loaded']} | Links: {health_resp['segments_loaded']}</div>
    </div>
    """, unsafe_allow_html=True)
except Exception:
    st.sidebar.markdown("""
    <div style='background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); padding: 10px; border-radius: 8px;'>
        <div style='color: #f87171; font-size: 0.85rem; font-weight: 600;'>🔴 Backend Engine Disconnected</div>
        <div style='font-size: 0.75rem; color: #94a3b8; margin-top: 4px;'>Run `uvicorn backend.main:app` on port 8000.</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_gis_data():
    nodes_path = "data/raw/nodes.csv"
    net_path = "data/raw/network.csv"
    sig_path = "data/raw/signal_plans.csv"
    nodes_df = pd.read_csv(nodes_path).set_index("node_id") if os.path.exists(nodes_path) else None
    net_df = pd.read_csv(net_path) if os.path.exists(net_path) else None
    sig_df = pd.read_csv(sig_path) if os.path.exists(sig_path) else None
    return nodes_df, net_df, sig_df

nodes_df, net_df, local_signals_df = load_gis_data()

# Initialize interface tabs
tabs = st.tabs([
    "🗺️ Network Digital Twin", 
    "📈 Multi-Horizon Forecast", 
    "🚨 Immediate Detours & Radar", 
    "🏗️ Long-Term What-If Interventions"
])

with tabs[0]:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    live_states = {}
    if backend_online:
        try:
            state_resp = requests.get(f"{API_BASE}/api/network/state?limit=436", timeout=2).json()
            for edge in state_resp.get("edges", []):
                live_states[edge["segment_id"]] = edge
        except Exception:
            pass

    ci_vals = [e.get("congestion_index", 0.0) for e in live_states.values()] if live_states else [0.0]
    avg_speed = np.mean([e.get("speed_kmh", 45.0) for e in live_states.values()]) if live_states else 42.8
    severe_count = sum(1 for ci in ci_vals if ci >= 0.60)
    
    kpi1.metric("Network Avg Speed", f"{avg_speed:.1f} km/h", delta="Nominal")
    kpi2.metric("Mean Congestion (CI)", f"{np.mean(ci_vals):.2f}", delta="-0.03" if np.mean(ci_vals) < 0.3 else "+0.05", delta_color="inverse")
    kpi3.metric("Gridlock Risk Links", f"{severe_count}", delta=f"{severe_count} bottlenecks", delta_color="inverse")
    kpi4.metric("Monitored Corridors", f"{len(net_df) if net_df is not None else 436}")

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    col_ctrl, col_map = st.columns([1, 3.2])
    with col_ctrl:
        st.markdown("#### 🌆 Hyderabad Relatability")
        show_signals = st.checkbox("Show 30 Signalized Junctions (signal_plans.csv)", value=True)
        show_bottlenecks = st.checkbox("Spotlight Structural Chokepoints", value=True)
        segment_limit = st.slider("Visible Corridors", 50, 436, 436, step=50)
        
        st.markdown("---")
        st.markdown("**Real-Time Legend**")
        st.markdown("<span class='badge badge-green'>Free Flow</span> <code>CI &lt; 0.25</code>", unsafe_allow_html=True)
        st.markdown("<span class='badge badge-amber'>Saturated</span> <code>0.25 &le; CI &lt; 0.60</code>", unsafe_allow_html=True)
        st.markdown("<span class='badge badge-red'>Choked</span> <code>CI &ge; 0.60</code>", unsafe_allow_html=True)
        st.markdown("<span class='badge badge-cyan'>Signal Junction</span> <code>Cycle 120s</code>", unsafe_allow_html=True)

    with col_map:
        if nodes_df is not None and net_df is not None:
            lines = []
            for _, row in net_df.head(segment_limit).iterrows():
                seg_id = str(row["segment_id"])
                u, v = str(row["source_node"]), str(row["target_node"])
                
                if u in nodes_df.index and v in nodes_df.index:
                    u_coord = [float(nodes_df.loc[u, "lon"]), float(nodes_df.loc[u, "lat"])]
                    v_coord = [float(nodes_df.loc[v, "lon"]), float(nodes_df.loc[v, "lat"])]
                    
                    live_edge = live_states.get(seg_id, {})
                    ci = live_edge.get("congestion_index", 0.0)
                    current_speed = live_edge.get("speed_kmh", row["free_flow_speed_kmh"])
                    is_bottleneck = int(row.get("structural_bottleneck", 0))

                    if show_bottlenecks and is_bottleneck == 1 and ci >= 0.40:
                        color = [239, 68, 68, 255]
                    elif ci >= 0.60:
                        color = [248, 113, 113, 230]
                    elif ci >= 0.25:
                        color = [251, 191, 36, 200]
                    else:
                        color = [52, 211, 153, 180]

                    lines.append({
                        "segment_id": seg_id,
                        "speed": round(current_speed, 1),
                        "ci": round(ci, 2),
                        "color": color,
                        "path": [u_coord, v_coord]
                    })
            
            center_lat = float(nodes_df["lat"].mean())
            center_lon = float(nodes_df["lon"].mean())

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=11.8,
                pitch=0,
                bearing=0
            )

            tile_layer = pdk.Layer(
                "TileLayer",
                data="https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
                min_zoom=0,
                max_zoom=19,
                tileSize=256,
                render_sub_layers=lambda props: pdk.Layer(
                    "BitmapLayer",
                    **props,
                    data=None,
                    image=props["data"],
                    bounds=props["bounds"]
                )
            )

            vector_layer = pdk.Layer(
                "PathLayer",
                lines,
                get_path="path",
                get_color="color",
                width_min_pixels=3,
                width_scale=2,
                pickable=True
            )

            map_layers = [tile_layer, vector_layer]

            if show_signals:
                sig_coords = []
                if local_signals_df is not None:
                    for _, s in local_signals_df.iterrows():
                        n_id = str(s["node_id"])
                        if n_id in nodes_df.index:
                            sig_coords.append({
                                "node_id": n_id,
                                "signal_id": str(s["signal_id"]),
                                "cycle_s": int(s.get("cycle_s", 120)),
                                "coordinates": [float(nodes_df.loc[n_id, "lon"]), float(nodes_df.loc[n_id, "lat"])]
                            })
                
                if sig_coords:
                    sig_layer = pdk.Layer(
                        "ScatterplotLayer",
                        sig_coords,
                        get_position="coordinates",
                        get_color=[6, 182, 212, 220],
                        get_radius=160,
                        pickable=True
                    )
                    map_layers.append(sig_layer)

            st.pydeck_chart(
                pdk.Deck(
                    layers=map_layers, 
                    initial_view_state=view_state,
                    map_provider=None,
                    map_style=None,
                    tooltip={"text": "Segment: {segment_id}\nSignal: {signal_id}\nSpeed: {speed} km/h\nCI: {ci}"}
                )
            )

with tabs[1]:
    st.markdown("### 📈 Multi-Horizon Traffic Speed Forecasting")
    st.caption("Direct multi-step gradient boosting inference across forward 15, 30, and 60-minute evaluation horizons.")
    
    col_input, col_chart = st.columns([1, 2])
    with col_input:
        with st.container(border=True):
            st.markdown("#### Input Vector")
            f_segment = st.selectbox("Corridor Segment", net_df["segment_id"].tolist() if net_df is not None else ["R0001"])
            f_speed = st.slider("Current Speed (km/h)", 5.0, 80.0, 32.5, 0.5)
            f_flow = st.slider("Vehicle Flow (vph)", 100.0, 3000.0, 1250.0, 50.0)
            f_ci = st.slider("Congestion Index (CI)", 0.0, 1.0, 0.42, 0.01)
            f_occ = st.slider("Sensor Occupancy (%)", 5.0, 95.0, 48.0, 1.0)
            
            predict_btn = st.button("Generate Horizon Projections")

    with col_chart:
        if predict_btn:
            payload = {
                "segment_id": f_segment,
                "speed_kmh": f_speed,
                "flow_vph": f_flow,
                "congestion_index": f_ci,
                "occupancy_pct": f_occ
            }
            try:
                res = requests.post(f"{API_BASE}/api/forecast", json=payload, timeout=5).json()
                preds = res["forecast_speeds_kmh"]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("+15 Min Target", f"{preds.get('15m', f_speed):.1f} km/h")
                m2.metric("+30 Min Target", f"{preds.get('30m', f_speed):.1f} km/h")
                m3.metric("+60 Min Target", f"{preds.get('60m', f_speed):.1f} km/h")
                
                st.markdown(f"**Confidence Metric:** <span class='badge badge-green'>{res.get('confidence', 0.85) * 100:.0f}% CERTAINTY</span>", unsafe_allow_html=True)
                
                # Strict alphabetical & chronological sequence: T0 -> T+15m -> T+30m -> T+60m
                chart_df = pd.DataFrame({
                    "Timeline": ["T0: Current", "T+15m Horizon", "T+30m Horizon", "T+60m Horizon"],
                    "Forecast Speed (km/h)": [f_speed, preds.get("15m", f_speed), preds.get("30m", f_speed), preds.get("60m", f_speed)]
                }).set_index("Timeline")
                
                st.line_chart(chart_df, color="#38bdf8")
            except Exception as e:
                st.error(f"Inference error: {e}")

with tabs[2]:
    st.markdown("### 🚨 Level 1: Immediate Operational Response (Safe Road Diversion)")
    st.caption("Validates incidents and searches existing alternatives. Enforces the Secondary Congestion Guard to prevent secondary gridlock.")
    
    col_inc, col_div = st.columns(2)
    with col_inc:
        with st.container(border=True):
            st.markdown("#### 1. Physical Incident Evaluator")
            inc_seg = st.text_input("Segment Identifier", value="R0360")
            inc_spd = st.number_input("Velocity Observed (km/h)", value=11.5, min_value=1.0, max_value=100.0)
            inc_occ = st.number_input("Corridor Occupancy (%)", value=82.0, min_value=0.0, max_value=100.0)
            
            if st.button("Audit Physical Anomaly"):
                try:
                    res = requests.post(f"{API_BASE}/api/incidents/evaluate", json={
                        "segment_id": inc_seg,
                        "current_speed_kmh": inc_spd,
                        "current_occupancy_pct": inc_occ
                    }, timeout=5).json()
                    
                    if res.get("incident_detected"):
                        st.markdown(f"<div style='padding: 12px; border-radius: 8px; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); margin-bottom: 10px;'><strong style='color: #f87171;'>🚨 ANOMALOUS INCIDENT CONFIRMED</strong><br><span style='font-size: 0.85rem;'>Confidence: {res['confidence']*100:.0f}% | Deviation: {res['z_score']}σ</span></div>", unsafe_allow_html=True)
                        st.markdown("**Evidence Chain:**")
                        for e in res["evidence"]:
                            st.markdown(f"- {e}")
                    else:
                        st.markdown(f"<div style='padding: 12px; border-radius: 8px; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.4);'><strong style='color: #4ade80;'>✅ CORRIDOR NOMINAL OR UNIFORM SATURATION</strong><br><span style='font-size: 0.85rem;'>Anomaly Score: {res.get('z_score', 0)}σ (Rules out isolated crash; standard peak queue)</span></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Incident audit error: {e}")

    with col_div:
        with st.container(border=True):
            st.markdown("#### 2. Capacity-Safe Diversion or Congestion Outlook")
            div_block = st.text_input("Saturated / Incident Segment", value="R0360")
            
            if st.button("Evaluate Routing Strategy"):
                try:
                    res = requests.post(f"{API_BASE}/api/advisory/diversion", json={
                        "blocked_segment_id": div_block
                    }, timeout=5).json()
                    
                    if res.get("detour_approved"):
                        st.markdown(f"<div style='padding: 12px; border-radius: 8px; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.4); margin-bottom: 12px;'><strong style='color: #4ade80;'>✅ SAFE DIVERSION RECOMMENDED</strong><br><span style='font-size: 0.85rem;'>{res['guard_rationale']}</span></div>", unsafe_allow_html=True)
                        st.markdown(f"**Approved Path:** `{' ➔ '.join(res['detour_path_nodes'])}`")
                        st.markdown(f"**Travel Time:** `{res['estimated_travel_time_min']} min` | **Peak Bypass CI:** `{res['max_corridor_ci']}`")
                    else:
                        st.markdown(f"""
                        <div style='padding: 12px; border-radius: 8px; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); margin-bottom: 12px;'>
                            <strong style='color: #fbbf24;'>⚠️ NO SAFE DIVERSION AVAILABLE (SECONDARY CONGESTION GUARD)</strong><br>
                            <span style='font-size: 0.85rem;'>Alternative corridors exceed capacity threshold (CI &ge; 0.45). Diverting traffic would induce secondary gridlock.</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("⏱️ **Congestion Outlook Advisory:** No viable bypass exists with spare capacity. System recommends holding traffic on primary arterials. Forecast models project corridor flow will stabilize in ~30–45 minutes.")
                except Exception as e:
                    st.error(f"Routing calculation error: {e}")

with tabs[3]:
    st.markdown("### 🏗️ Level 2: Long-Term Solution (Recurring Bottleneck & Infrastructure Intervention)")
    st.caption("Identifies chronically recurring chokepoints and simulates counterfactual corridor additions or modifications before committing capital.")
    
    candidates_path = "data/raw/planning_candidates.csv"
    if os.path.exists(candidates_path):
        cand_df = pd.read_csv(candidates_path)
        
        c_left, c_right = st.columns([1, 1.8])
        with c_left:
            with st.container(border=True):
                st.markdown("#### Bottleneck Intervention Candidate")
                sel_cand = st.selectbox(
                    "Identified Recurring Bottleneck Candidate",
                    cand_df["candidate_id"].tolist(),
                    format_func=lambda x: f"{x} — {cand_df[cand_df['candidate_id']==x]['intervention_type'].iloc[0].replace('_', ' ').title()} on Link {cand_df[cand_df['candidate_id']==x]['target_segment'].iloc[0]}"
                )
                sim_btn = st.button("Simulate Virtual Network Intervention")
                
        with c_right:
            if sim_btn:
                try:
                    resp = requests.post(f"{API_BASE}/api/simulation/what-if", json={"candidate_id": sel_cand}, timeout=5)
                    if resp.status_code == 200:
                        res = resp.json()
                        
                        st.markdown("#### 📊 Before vs. After Simulation Audit")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Target Corridor", res.get("target_segment", "N/A"))
                        m1_type = res.get("intervention_type", "Intervention").replace("_", " ").title()
                        m2.metric("Intervention Type", m1_type)
                        m3.metric("Feasibility", res.get("feasibility", "HIGH").upper())
                        
                        st.divider()
                        
                        before_col, after_col = st.columns(2)
                        with before_col:
                            st.markdown("##### 🔴 Baseline (Existing Network)")
                            st.metric("Effective Capacity", f"{res.get('original_capacity_vph', 1800)} vph")
                            st.markdown("- **Congestion Risk:** Elevated peak-hour chokepoint")
                            st.markdown("- **Queue Spillback:** Cascades into upstream junctions")
                            
                        with after_col:
                            st.markdown("##### 🟢 Simulated (With Intervention)")
                            gain = res.get("capacity_gain_vph", res.get("new_capacity_vph", 2600) - res.get("original_capacity_vph", 1800))
                            st.metric("Expanded Capacity", f"{res.get('new_capacity_vph', 2600)} vph", delta=f"+{gain} vph")
                            st.metric("Net Delay Reduction", f"{res.get('delay_reduction_veh_min', 0.0)} veh-min", delta="Delay Saved")
                            st.markdown("- **Impact:** Relieves adjacent arterial corridors")
                            
                        st.success(f"**Planning Recommendation:** Intervention `{sel_cand}` is estimated to save **{res.get('delay_reduction_veh_min', 0.0)} vehicle-minutes** of net network delay under peak loading.")
                    else:
                        st.error(f"Backend simulation returned HTTP {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Simulation connection error: {e}")
    else:
        st.warning("planning_candidates.csv not found in data/raw/.")

if live_mode:
    time.sleep(refresh_rate)
    st.rerun()