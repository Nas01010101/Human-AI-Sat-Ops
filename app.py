"""
ORBIT-GUARD — Satellite Collision Decision Support

Research platform comparing AI interface designs for human-in-the-loop
collision avoidance decision-making.
"""

import streamlit as st

st.set_page_config(
    page_title="ORBIT-GUARD",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Title ────────────────────────────────────────────────────────
st.title("ORBIT-GUARD")
st.markdown(
    "**Satellite Collision Decision Support** — "
    "A research platform for studying human-AI teaming in conjunction assessment."
)

st.divider()

# ── Overview Metrics ─────────────────────────────────────────────
try:
    from src.conjunction_data import load_conjunctions
    events = load_conjunctions()
    active_count = len(events)
    critical_count = sum(1 for e in events if e.get("risk_level") == "critical")
    high_count = sum(1 for e in events if e.get("risk_level") == "high")
    closest_km = min(e.get("min_range_km", 999) for e in events) if events else 0
except Exception:
    active_count, critical_count, high_count, closest_km = 0, 0, 0, 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conjunction Events", f"{active_count:,}")
c2.metric("Critical Risk", critical_count)
c3.metric("High Risk", high_count)
c4.metric("Closest Approach", f"{closest_km:.3f} km")

st.divider()

# ── Navigation ───────────────────────────────────────────────────
st.subheader("Modules")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Collision Dashboard**")
    st.caption("Visualize conjunction data, filter by risk level, and inspect individual events.")
    st.page_link("pages/1_Collision_Dashboard.py", label="Open Dashboard →")

with col2:
    st.markdown("**Triage Cockpit**")
    st.caption("Simulate operator decision-making across three AI interface variants.")
    st.page_link("pages/2_Triage_Cockpit.py", label="Open Cockpit →")

with col3:
    st.markdown("**Simulation Results**")
    st.caption("Monte Carlo analysis of accuracy, miss rate, and operational cost.")
    st.page_link("pages/3_Simulation.py", label="View Results →")

with col4:
    st.markdown("**Methodology**")
    st.caption("Data sources, model architecture, processing pipeline, and limitations.")
    st.page_link("pages/4_Methodology.py", label="Read Details →")

st.divider()

# ── About ────────────────────────────────────────────────────────
st.subheader("About This Project")

st.markdown("""
This platform investigates how AI transparency and cognitive ergonomics 
affect operator performance in satellite collision avoidance triage.

**Research question:** Does adding AI evidence panels, confidence scores, and 
safety interlocks improve decision accuracy — especially for novice or fatigued 
controllers?

**Pipeline:**
1. Ingest real conjunction data from CelesTrak SOCRATES (114,000+ events/day)
2. Train a Random Forest classifier to predict collision risk levels
3. Simulate operator triage across three UI variants and three operator profiles
4. Measure decision accuracy, critical miss rate, and operational cost

**Data:** All conjunction records are sourced from the public 
[CelesTrak SOCRATES](https://celestrak.org/SOCRATES/) catalog.
Risk labels are derived from physics-based thresholds (miss distance + collision probability),
not from human annotation.
""")

st.caption("Data provided by CelesTrak. This is a research prototype — not for operational use.")
