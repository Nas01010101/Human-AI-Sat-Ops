"""
Simulation Results — Monte Carlo analysis of operator performance.
"""

import time
import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TABLES_DIR

st.set_page_config(page_title="Simulation Results", page_icon="📊", layout="wide")

st.title("Simulation Results")
st.caption("Monte Carlo comparison of operator performance across interface variants.")

# ── Load ─────────────────────────────────────────────────────────
results_path = os.path.join(TABLES_DIR, "simulation_results.csv")

if not os.path.exists(results_path):
    st.info("No simulation results found. Run the pipeline to generate data.")

    if st.button("Run Pipeline"):
        from src.run_all import run_pipeline
        with st.spinner("Running simulation (this may take ~1 minute)..."):
            try:
                run_pipeline()
                st.success("Done.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
    st.stop()

try:
    df = pd.read_csv(results_path)
except Exception:
    st.error("Could not read simulation results. Re-run the pipeline.")
    st.stop()

# ── Summary ──────────────────────────────────────────────────────
best_variant = df.groupby("variant")["cost_mean"].mean().idxmin()
worst_variant = df.groupby("variant")["cost_mean"].mean().idxmax()
cost_reduction = (
    df.groupby("variant")["cost_mean"].mean().max() -
    df.groupby("variant")["cost_mean"].mean().min()
) / df.groupby("variant")["cost_mean"].mean().max()

c1, c2, c3 = st.columns(3)
c1.metric("Best Variant", f"Mode {best_variant}")
c2.metric("Cost Reduction", f"{cost_reduction:.1%}")
c3.metric("Simulation Runs", f"{len(df)}")

st.divider()

# ── Charts ───────────────────────────────────────────────────────
colors = {"A": "#8c8c8c", "B": "#1f77b4", "C": "#2ca02c"}

tab1, tab2, tab3 = st.tabs(["Decision Accuracy", "Critical Miss Rate", "Operational Cost"])

with tab1:
    fig = px.bar(
        df, x="profile", y="accuracy_mean", color="variant", barmode="group",
        error_y="accuracy_std",
        color_discrete_map=colors,
        labels={"accuracy_mean": "Decision Accuracy", "profile": "Operator Profile", "variant": "Variant"},
    )
    fig.update_layout(margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher accuracy = better adherence to correct collision response SOPs.")

with tab2:
    fig = px.bar(
        df, x="profile", y="miss_rate_mean", color="variant", barmode="group",
        color_discrete_map=colors,
        labels={"miss_rate_mean": "Critical Miss Rate", "profile": "Operator Profile", "variant": "Variant"},
    )
    fig.update_layout(margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Miss rate = proportion of critical/high events that were incorrectly dismissed.")

with tab3:
    fig = px.bar(
        df, x="profile", y="cost_mean", color="variant", barmode="group",
        error_y="cost_std",
        color_discrete_map=colors,
        labels={"cost_mean": "Operational Cost", "profile": "Operator Profile", "variant": "Variant"},
    )
    fig.update_layout(margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cost function: missed critical (×50), missed high (×20), false maneuver (×8), false prepare (×3).")

# ── Raw Data ─────────────────────────────────────────────────────
st.divider()
with st.expander("Raw simulation data"):
    st.dataframe(df, use_container_width=True)
