"""
Collision Dashboard — Conjunction event visualization and risk analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CACHE_DIR
from src.conjunction_data import load_conjunctions

st.set_page_config(page_title="Collision Dashboard", page_icon="🛰️", layout="wide")

st.title("Collision Dashboard")
st.caption("Source: CelesTrak SOCRATES • AI risk predictions from Random Forest classifier")

# ── Load Data ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    path = os.path.join(CACHE_DIR, "conjunctions.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return load_conjunctions()

try:
    events = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if not events:
    st.warning("No conjunction events available. Run the pipeline first: `python src/run_all.py`")
    st.stop()

df = pd.DataFrame(events)

# ── Summary Metrics ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Events", len(df))
c2.metric("Critical", len(df[df["risk_level"] == "critical"]))
c3.metric("High", len(df[df["risk_level"] == "high"]))
c4.metric("Closest Approach", f"{df['min_range_km'].min():.3f} km")

st.divider()

# ── Risk Scatter Plot ────────────────────────────────────────────
st.subheader("Risk Landscape")

fig = px.scatter(
    df,
    x="min_range_km",
    y="max_probability",
    color="risk_level",
    size="time_to_tca_hours",
    hover_data=["name_1", "name_2", "relative_speed_kms"],
    log_y=True,
    labels={
        "min_range_km": "Minimum Range (km)",
        "max_probability": "Collision Probability",
        "time_to_tca_hours": "Hours to TCA",
        "risk_level": "Risk Level"
    },
    color_discrete_map={
        "critical": "#d62728", "high": "#ff7f0e",
        "medium": "#bcbd22", "low": "#2ca02c"
    },
    height=450,
)

fig.update_layout(
    xaxis=dict(showgrid=True, gridcolor="#eee"),
    yaxis=dict(showgrid=True, gridcolor="#eee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30),
)

st.plotly_chart(fig, use_container_width=True)

# ── Filters + Data Table ─────────────────────────────────────────
st.subheader("Event Feed")

col_f, col_t = st.columns([1, 3])

with col_f:
    filter_risk = st.multiselect(
        "Risk Level",
        ["critical", "high", "medium", "low"],
        default=["critical", "high", "medium"]
    )
    max_range = st.slider("Max Range (km)", 0.1, 50.0, 25.0)

with col_t:
    filtered = df[
        (df["risk_level"].isin(filter_risk)) &
        (df["min_range_km"] <= max_range)
    ].sort_values("max_probability", ascending=False)

    st.dataframe(
        filtered[["event_id", "risk_level", "name_1", "name_2",
                   "min_range_km", "max_probability", "time_to_tca_hours",
                   "action_true"]].head(100),
        column_config={
            "event_id": st.column_config.TextColumn("Event ID"),
            "risk_level": st.column_config.TextColumn("Risk"),
            "min_range_km": st.column_config.NumberColumn("Miss Dist (km)", format="%.3f"),
            "max_probability": st.column_config.NumberColumn("Probability", format="%.2e"),
            "time_to_tca_hours": st.column_config.NumberColumn("Hours to TCA", format="%.1f"),
            "action_true": st.column_config.TextColumn("Recommended Action"),
        },
        use_container_width=True,
        height=400,
    )

# ── 3D Encounter View ────────────────────────────────────────────
st.subheader("Encounter Geometry")

if not filtered.empty:
    selected_id = st.selectbox(
        "Select event",
        filtered["event_id"].head(20)
    )
    evt = filtered[filtered["event_id"] == selected_id].iloc[0]

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text",
        marker=dict(size=8, color="#1f77b4"),
        text=[evt["name_1"]], textposition="top center",
        name="Primary"
    ))

    r = evt["min_range_km"]
    v = np.array([1.0, 0.5, 0.2])
    v = v / np.linalg.norm(v) * r

    fig3.add_trace(go.Scatter3d(
        x=[v[0]], y=[v[1]], z=[v[2]], mode="markers+text",
        marker=dict(size=8, color="#d62728"),
        text=[evt["name_2"]], textposition="top center",
        name="Secondary"
    ))

    fig3.update_layout(
        scene=dict(
            xaxis_title="Radial (km)",
            yaxis_title="In-Track (km)",
            zaxis_title="Cross-Track (km)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
    )
    st.plotly_chart(fig3, use_container_width=True)

    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("TCA", evt["tca"])
    col_d2.metric("Range", f"{evt['min_range_km']:.3f} km")
    col_d3.metric("Probability", f"{evt['max_probability']:.2e}")
