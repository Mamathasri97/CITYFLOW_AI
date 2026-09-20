# 🚦 CITYFLOW AI
### Urban Traffic Flow & Incident Intelligence System
**NeuraX Hackathon 3.0 — AI in Smart Cities ***

CITYFLOW AI is a software-only, simulated urban traffic intelligence and decision-support platform designed for municipal traffic management centers and urban planners.
It transforms road-network topology and traffic observations into an active spatial-temporal graph, detects abnormal incidents, forecasts traffic conditions 15, 30, and 60 minutes ahead, recommends capacity-safe diversions, and evaluates long-term infrastructure interventions through what-if simulations.
CITYFLOW AI is not designed as an individual navigation or route-planning application. Its purpose is to provide network-level decision support for understanding traffic conditions, anticipating congestion propagation, recommending simulated operational responses, and evaluating longer-term network improvements.

### Checkpoint 1 — Evaluation Summary

CITYFLOW AI is aligned with the three Checkpoint 1 evaluation areas:

Evaluation Area| Where It Is Covered
Problem Understanding — 5 Marks| Problem Statement, Our Solution, Key Objectives
System Architecture — 5 Marks| How CITYFLOW AI Works, System Architecture
Technical Approach — 5 Marks| AI/ML Pipeline, Congestion Detection, Incident Detection, Traffic Forecasting, Adaptive Recommendations, Infrastructure What-If Simulator

Problem Understanding

CITYFLOW AI addresses network-level urban traffic management, not individual navigation. It analyzes congestion, abnormal traffic behavior, incident propagation, future traffic conditions, and recurring bottlenecks to support simulated operational and long-term planning decisions.

System Architecture

The system converts organizer-provided road-network and traffic datasets into a spatiotemporal road graph, processes traffic features, performs congestion/incident detection and multi-horizon forecasting, and passes the results to a decision engine for diversion advisories and infrastructure what-if analysis.

Technical Approach

The system combines graph-based traffic modeling, congestion indexing, statistical and spatial incident detection, 15/30/60-minute traffic forecasting, capacity-aware diversion logic, and infrastructure what-if simulation. All recommendations remain simulated and advisory.
---

## 📌 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Our Solution](#-our-solution)
3. [Key Objectives](#-key-objectives)
4. [Key Features](#-key-features)
5. [How CITYFLOW AI Works](#-how-cityflow-ai-works)
6. [System Architecture](#️-system-architecture)
7. [AI/ML Pipeline](#-aiml-pipeline)
8. [Congestion Detection](#-congestion-detection)
9. [Incident Detection](#-incident-detection)
10. [Traffic Forecasting](#-traffic-forecasting)
11. [Adaptive Recommendations](#-adaptive-recommendations)
12. [Infrastructure What-If Simulator](#️-infrastructure-what-if-simulator)
13. [Explainability & Confidence](#-explainability--confidence)
14. [Technology Stack](#️-technology-stack)
15. [Dataset](#-dataset)
16. [Project Structure](#-project-structure)
17. [Installation](#️-installation)
18. [Running the Application](#️-running-the-application)
19. [API Endpoints](#-api-endpoints)
20. [Evaluation](#-evaluation)
21. [Why CITYFLOW AI?](#-why-cityflow-ai)
22. [Limitations](#️-limitations)
23. [Future Scope](#-future-scope)
24. [Team & Vision](#-team--vision)

---

## 🚨 Problem Statement
Rapidly expanding cities such as Hyderabad experience complex traffic conditions caused by:
* Peak-hour commuter surges
* Road-network bottlenecks and localized geometric capacity limits
* Uneven traffic distribution across flyovers and arterials
* Localized incidents and disruptions
* Queue spillback across neighboring road segments
* Weather-related slowdowns and event surges
* Limited visibility into downstream propagation effects
* Reactive traffic-management decisions

Traditional traffic management often responds after congestion has already propagated across nearby intersections and corridors.
CITYFLOW AI addresses this problem by providing a simulated intelligence layer capable of:
* Monitoring network conditions across continuously updated analysis cycles.
* Detecting abnormal traffic incidents while suppressing false alarms
* Forecasting future traffic conditions 15 to 60 minutes ahead
* Identifying potential queue spillback before gridlock occurs
* Generating tactical, load-balanced diversion advisories
* Identifying recurring structural bottlenecks
* Simulating infrastructure improvements (flyovers, lane expansions)

---

## 💡 Our Solution
CITYFLOW AI operates across two decision horizons:

### ⚡ Tactical Horizon — Current state to 60 Minutes ahead
The system analyzes current traffic conditions and road topology to:
* Calculate normalized congestion severity per road segment.
* Detect abnormal incidents by cross-validating speed anomalies with upstream/downstream observations in the organizer-provided dataset.
* Forecast segment speeds at 15-, 30-, and 60-minute windows.
* Identify possible spillback vectors along connecting links.
* Generate alternative routes with virtual capacity penalties.
* Reject candidate diversions that could create secondary congestion on bypass corridors.

### 🏗️ Strategic Horizon — Long-Term Planning
The system analyzes historical traffic behavior to:
* Identify persistent recurring bottlenecks across peak-hour cycles.
* Separate structural geometric problems from transient incidents.
* Simulate virtual lane additions, flyovers, and elevated corridors.
* Simulate junction grade separation and one-way network modifications.
* Quantify network delay reduction before capital allocation.

---

## 🎯 Key Objectives
* Build an active directed road graph $G = (V, E)$ from road network datasets.
* Calculate normalized Congestion Index ($CI$) values continuously across all links.
* Detect anomalous incidents using spatial and temporal evidence while suppressing false alarms.
* Forecast traffic speeds at 15, 30, and 60-minute horizons, with evaluation on unseen traffic conditions.
* Generate capacity-aware diversion recommendations that avoid secondary gridlock.
* Identify persistent infrastructure bottlenecks via peak recurrence profiling.
* Simulate structural road-network modifications and output before/after travel-time deltas.
* Provide explainable evidence and confidence estimates based on model uncertainty and input completeness.

---

## ✨ Key Features
1. **🗺️ Spatiotemporal Graph Modeling:** The road network is modeled as a directed graph where each edge contains link ID, terminal nodes, segment length, lane count, free-flow speed, observed speed, flow, and spatial adjacency.
2. **🚨 Dual-Filter Incident Detection:** Combines statistical velocity drop tracking ($Z \ge 2.5$) with upstream queue growth and downstream starvation analysis to rule out ordinary rush-hour saturation.
3. **📈 Multi-Horizon Traffic Forecasting:** Direct multi-step regression models forecasting continuous link velocities at $t+15$, $t+30$, and $t+60$ minutes.
4. **🛣️ Gridlock-Safe Diversion:** Evaluates candidate detours using penalized shortest paths; automatically rejects bypasses that lack sufficient capacity headroom ($CI_{\text{detour}} \ge 0.45$).
5. **🏗️ Infrastructure What-If Simulator:** Quantifies network delay deltas before and after hypothetical modifications (lane additions, elevated bypasses, signal re-timing).
6. **🔍 Explainable AI:** Outputs evidence rationales alongside confidence scores reflecting model uncertainity and input completeness
---

## 🔄 How CITYFLOW AI Works


                  Traffic & Network Data
                            │
                            ▼
               ┌─────────────────────────┐
               │   Data Preprocessing    │
               │  - Missing Imputation   │
               │  - Temporal Windowing   │
               └────────────┬────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │ Road Graph Construction │
               │ NetworkX Directed Graph │
               └────────────┬────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │   Feature Engineering   │
               │ Lags + Spatial Features │
               │   + Temporal Context    │
               └───────┬─────────┬───────┘
                       │         │
                       ▼         ▼
               ┌────────────┐ ┌──────────────┐
               │  Incident  │ │ Forecasting  │
               │ Detection  │ │ 15/30/60 min │
               └─────┬──────┘ └──────┬───────┘
                     │               │
                     └───────┬───────┘
                             ▼
               ┌──────────────────────────┐
               │ Decision & Optimization  │
               │ Routing + Capacity Check │
               │   What-If Simulation     │
               └────────────┬─────────────┘
                            │
                            ▼
               ┌──────────────────────────┐
               │   Operations Dashboard   │
               │ Maps + Alerts + Forecast │
               │     Recommendations      │
               └──────────────────────────┘


---

## 🏛️ System Architecture

                     ORGANIZER DATASETS
                       │
              ┌────────┴────────┐
              ↓                 ↓
       Road Network Data    Traffic Data
              │                 │
              └────────┬────────┘
                       ↓
              DATA PREPROCESSING
                       ↓
           SPATIOTEMPORAL FEATURES
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
     Congestion    Incident      Forecasting
     Detection     Detection     15/30/60 min
          │            │            │
          └────────────┼────────────┘
                       ↓
               DECISION ENGINE
                /           \
               ↓             ↓
        Diversion       What-If
        Advisory       Simulation
               \             /
                ↓           ↓
                 OPERATIONS
                  DASHBOARD



---

## 🤖 AI/ML Pipeline

Data Ingestion ──► Missing Imputation ──► Graph Features ──► Speed Lags ──► Model Training ──► Inference & Policy


### Input Features per Segment

* **Temporal Velocity Lags:** $v_e(t)$, $v_e(t-5)$, $v_e(t-15)$, $v_e(t-30)$
* **Spatial Neighbor Metrics:** Upstream mean velocity $\bar{v}_{N_{in}(e)}(t)$, Downstream mean velocity $\bar{v}_{N_{out}(e)}(t)$
* **Contextual Features:** Hour of day, Day of week, Peak-hour boolean flag
* **State Metric:** Congestion Index $CI_e(t)$

### Forecasting Model Architecture

Independent direct Gradient Boosted Decision Tree models (Gradient-Boosted Decision Tree model) are trained for each forecasting horizon:

* **Model 1:** $t + 15\text{ minutes}$

* **Model 2:** $t + 30\text{ minutes}$

* **Model 3:** $t + 60\text{ minutes}$


*Direct multi-step forecasting eliminates the recursive accumulation of prediction errors over extended prediction windows.*

---

## 🚦 Congestion Detection

The Congestion Index ($CI_e$) normalizes current velocity against segment free-flow speed:

$$CI_e(t) = \max\left(0, \min\left(1, \frac{v_f - v_e(t)}{v_f}\right)\right)$$

| Congestion Index ($CI$) | Classification | Operational Traffic Condition |
| --- | --- | --- |
| **$< 0.25$** | **Free Flow** | Speeds nominal; optimum throughput; minimal delay |
| **$0.25 \le CI < 0.60$** | **Moderate Congestion** | Density rising; volume near capacity; minor queuing |
| **$\ge 0.60$** | **Severe Gridlock Risk** | Stop-and-go conditions; high probability of spillback propagation

 |

---

## 🚨 Incident Detection

To avoid classifying normal rush-hour slowdowns as incidents, CITYFLOW AI validates three independent signals:


Normal Peak Congestion:    Uniform corridor-wide speed drop ────► Flagged as Recurring Peak (No Incident Alert)
Real Physical Incident:    Localized Drop + Upstream Queue + Downstream Starvation ────► High-Priority Incident Alert


1. **Statistical Anomaly:** Speed drop exceeding $Z \ge 2.5$ standard deviations below the historical time-of-day mean:

$$Z_e(t) = \frac{\mu_{e, \text{hist}} - v_e(t)}{\sigma_{e, \text{hist}}}$$


2. **Spatial Discrepancy Verification:** True incidents produce severe deceleration on link $e$ while downstream link $e+1$ experiences starvation (low density, normal speeds) and upstream link $e-1$ accumulates queue spillback.


3. **Temporal Filtering:** Single-timestamp sensor noise is filtered via an exponential moving window before raising alarms.



---

## 📊 Traffic Forecasting

Continuous velocity prediction across multi-horizon steps:


$$\hat{v}_e(t+k) = f_k\left(\mathbf{X}_e(t)\right), \quad k \in \{15, 30, 60\text{ minutes}\}$$


┌────────────────────────────────────────┐
│ Corridor Segment: Arterial Link 14     │
├────────────────────────────────────────┤
│ Current Speed:            22.4 km/h    │
│ Forecast (+15 min):       18.1 km/h    │
│ Forecast (+30 min):       14.2 km/h    │
│ Forecast (+60 min):       11.0 km/h    │
│ Trend / Risk:             Severe Spillback Predicted  │
└────────────────────────────────────────┘



This enables municipal operators to initiate mitigation policies up to 60 minutes before gridlock sets in.

---

## 🧭 Adaptive Recommendations

### Advisory Level 1 — Simulated Signal-Timing Advisory

Reallocates green-phase timing at downstream signalized junctions to accelerate queue dissipation from critical bottlenecks.The system does not communicate with or control physical traffic signals.

### Advisory Level 2 — Tactical Corridor Diversions

Calculates alternate bypass corridors using graph-based routing with dynamic virtual travel-time penalties applied to saturated segments.


Incident / Bottleneck Detected
               ↓
    Generate Candidate Paths
               ↓
 Apply Virtual Congestion Penalties
               ↓
  Verify Downstream Capacity Headroom
               ↓
       Is Bypass CI < 0.45?
           ↙        ↘
        YES          NO
         ↓            ↓
  Approve Detour   Reject Detour (Prevent Secondary Gridlock)



---

## 🏗️ Infrastructure What-If Simulator

Structural bottlenecks are isolated by evaluating historical recurrence:


$$\text{Recurrence Ratio} = \frac{\text{Periods where } CI_e \ge 0.65}{\text{Total Peak Observation Periods}} \ge 0.40$$

Traffic engineers interactively simulate interventions on persistent choke points:

* Virtual lane capacity expansion (simulating road widening)


* Grade-separated elevated corridors / flyovers


* Junction gyratories and one-way loop conversions



### Network Delay Evaluation

Quantifies the before-and-after performance impact across the network:


$$\Delta \text{Network Delay} = \sum_{e \in E} \left( T_{\text{post}, e} - T_{\text{pre}, e} \right) \times \text{Volume}_e$$

---

## 🔎 Explainability & Confidence

Every alert and recommendation includes human-readable evidence:

```text
illustrative dashboard output: 
[ALERT] Incident Detected on Link 14 (Confidence: 89%)
├── Evidence 1: Current speed (17.2 km/h) dropped 64% below historical mean (48.0 km/h)
├── Evidence 2: Z-score anomaly score = 2.82σ
├── Evidence 3: Downstream Link 15 shows volume starvation (Normal speed, low count)
└── Evidence 4: Upstream Link 13 exhibits queue spillback propagation (+42% occupancy)

```

Confidence scores are scaled based on input completeness:


The system reports a confidence score based on model uncertainty and input completeness. Confidence decreases when uncertainty or missing data increases


---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| **Language** | Python 3.10+ | Core data processing and ML pipeline |
| **API Framework** | FastAPI, Uvicorn, Pydantic | REST service endpoints and payload validation |
| **Graph Modeling** | NetworkX | Directed spatial network representation and routing |
| **Data & ML** | Pandas, NumPy, Scikit-learn, LightGBM | Preprocessing, feature engineering, and forecasting |
| **Dashboard** | Streamlit, PyDeck / Folium | Real-time map visualization, alerts, and what-if console

 |
| **Data Formats** | CSV / JSON | Network topology and observation time series |

---

## 📂 Dataset

CITYFLOW AI ingests organizer-provided traffic and road-network datasets:

* **Road Network Layer:** Graph topology defining intersections (nodes) and road segments (directed edges), segment lengths, lane counts, free-flow speeds, and connectivity.
* **Traffic Observation Layer:** Time-series records containing timestamps, segment/link identifiers, average vehicle velocities, and flow volumes.
* **Validation Split:** Partitioned historical observations reserved for evaluating 15–60 minute forecast error (MAE/RMSE) and incident detection precision/recall.



---

## 📁 Project Structure


CITYFLOW-AI/
├── README.md                       # Complete Project Documentation
├── requirements.txt                # Python Dependencies
├── backend/
│   ├── main.py                     # FastAPI REST API Server
│   ├── graph/
│   │   ├── network_builder.py      # NetworkX Road Graph Construction
│   │   └── routing_engine.py       # Diversion Engine & Secondary Guard
│   ├── models/
│   │   ├── forecasting.py          # 15, 30, 60-min Speed Regressors
│   │   └── incident_detector.py    # Spatial Anomaly & Noise Filters
│   └── simulation/
│       └── what_if_engine.py       # Infrastructure Scenario Evaluator
├── frontend/
│   └── app.py                      # Streamlit Operations Dashboard
└── data/
    ├── raw/                        # Organizer-provided datasets
    └── processed/                  # Computed feature matrices & graphs


---

## ⚙️ Installation

### 1. Clone or Open the Repository

```bash
cd CITYFLOW-AI

```

### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## ▶️ Running the Application

### 1. Launch FastAPI Backend

From the project root:

```bash
uvicorn backend.main:app --reload --port 8000

```

* API Root: `http://localhost:8000`
* Interactive API Docs (Swagger): `http://localhost:8000/docs`

### 2. Launch Streamlit Operations Console

In a second terminal window (with virtual environment activated):

```bash
streamlit run frontend/app.py

```

* Operations Console: `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | System health check and status |
| `GET` | `/api/network/state` | Returns current speeds, congestion index, and link travel times |
| `POST` | `/api/incidents/evaluate` | Evaluates edge speed drop against historical mean/std for incidents |
| `POST` | `/api/forecast` | Returns 15, 30, and 60-minute forward speed forecasts and confidence

 |
| `POST` | `/api/advisory/diversion` | Recommends capacity-checked bypass routes avoiding saturated links |

---

## 🧪 Evaluation

| Capability | Metric |
|---|---|
| Incident Detection | Precision, Recall, F1 |
| False Alarm Control | False Alarm Rate |
| Traffic Forecasting | MAE, RMSE |
| Diversion Recommendations | Estimated delay reduction / capacity safety |
| Infrastructure What-If | Network travel-time/delay change |
| Robustness | Performance under noisy/missing data |

Target values will be reported after evaluation on the organizer-provided validation/test sequences.

---

## 🆚 Why CITYFLOW AI?

```text
Traditional Navigation Apps:        Optimizes individual routes (Induces neighborhood rat-running)
Generic Conversational Bots:       Provides static text explanations (No spatial graph reasoning)
CITYFLOW AI (Decision Engine):      System-wide network equilibrium, spillback forecasting, 
                                    capacity-safe diversions, and long-term what-if infrastructure ROI.

```

---

## ⚠️ Limitations

* **Simulated Advisory Scope:** All generated diversion policies and signal split advisories are simulated decision-support recommendations; no live signal control, camera access, or physical roadside hardware control is required or permitted.All outputs are advisory and simulated. The system does not control real-world signals, access cameras, GPS devices, roadside sensors, municipal infrastructure, or initiate physical construction.
* **Network Coverage:** Accuracy relies on the topological coverage of monitored links; unmonitored roads are interpolated via spatial neighbor averages.
* **Driver Compliance Assumption:** Diversion simulations model a realistic 30–50% driver compliance rate rather than 100% deterministic compliance.

---

## 🚀 Future Scope

* Multi-agent reinforcement learning for dynamic signal cycle optimization across arterial corridors.
* Ingestion of public bus telemetry to prioritize high-occupancy transit corridors.
* CCTV edge integration for automated optical flow extraction where live camera feeds are available.

---


## 👥 Team & Vision

* **Competition:** NeuraX Hackathon 3.0


* **Track:** Urban Traffic Flow & Incident Intelligence


* **Vision:** *Transforming reactive municipal traffic control into predictive, evidence-based urban mobility intelligence.*
