# ORBIT-GUARD — Human-AI Decision Support for Satellite Collision Avoidance

A research platform that studies how AI interface design affects human operator performance in satellite conjunction triage. Built with real orbital data from CelesTrak SOCRATES.

![Python](https://img.shields.io/badge/python-3.11-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red) ![License](https://img.shields.io/badge/license-MIT-green)

## What It Does

Satellite operators must triage hundreds of close-approach warnings per day. This platform asks: **does the way AI presents its recommendations change how well operators make decisions?**

It compares three UI variants across three operator cognitive profiles using Monte Carlo simulation on real conjunction data:

| Variant | Design | Key Feature |
|---------|--------|-------------|
| A | Basic Alert List | Chronological feed, AI recommendation only |
| B | AI-Ranked + Evidence | Collapsible evidence panels, confidence scores |
| C | Focus Mode | Severity prioritization, SOP checklists, safety interlocks |

| Profile | Description |
|---------|-------------|
| Novice | High automation reliance, low verification rate |
| Experienced | Calibrated trust, pattern recognition |
| Fatigued | Rising error rate, checklist skipping |

## Architecture

```
app.py                      # Streamlit home — overview metrics
pages/
  1_Collision_Dashboard.py  # Risk scatter, event table, 3D encounter view
  2_Triage_Cockpit.py       # Interactive operator triage (3 UI variants)
  3_Simulation.py           # Monte Carlo results visualization
  4_Methodology.py          # Data sources, model, design, limitations
src/
  conjunction_data.py       # SOCRATES fetch → parse → enrich pipeline
  risk_model.py             # Random Forest classifier + explanation generator
  operator_sim.py           # Cognitive operator model + Monte Carlo engine
  config.py                 # Thresholds, profiles, cost weights
  run_all.py                # Full pipeline: data → model → sim → export
```

## Quick Start

```bash
git clone https://github.com/Nas01010101/Human-AI-Sat-Ops.git
cd Human-AI-Sat-Ops
pip install -r requirements.txt

# Run full pipeline (fetches live data, trains model, runs simulation ~60s)
python src/run_all.py

# Launch the dashboard
streamlit run app.py
```

Open http://localhost:8501

## Pipeline

1. **Data** — Fetches live conjunction data from [CelesTrak SOCRATES](https://celestrak.org/SOCRATES/) (130k+ events). Falls back to cached/synthetic data offline.
2. **ML** — Trains a Random Forest on 8 physics-derived features (miss distance, collision probability, relative velocity, data staleness, etc.) with 5-fold stratified CV.
3. **Simulation** — 500 Monte Carlo runs per condition (3 profiles × 3 variants = 9 conditions). Each run simulates a full operator shift with attention, fatigue, verification, and automation-bias mechanics.
4. **Metrics** — Decision accuracy, critical miss rate, false alarm rate, and weighted operational cost (missed critical ×50, missed high ×20, false maneuver ×8).

## Key Design Decisions

- **Polling-based cache** (6h TTL) avoids hammering the SOCRATES API
- **Physics-based labels** (not human annotation) — risk level derived from miss distance + collision probability thresholds aligned with standard conjunction assessment practice
- **Mechanistic operator model** — not a black-box; each parameter (verification propensity, fatigue growth, checklist adherence) maps to a documented human factors construct
- **Variant C interlock** — the "Execute Maneuver" button is disabled unless AI predicts critical risk, simulating a safety gate

## Data Source

All conjunction records from [CelesTrak SOCRATES](https://celestrak.org/SOCRATES/) (T.S. Kelso). This is a research prototype — not for operational use.
