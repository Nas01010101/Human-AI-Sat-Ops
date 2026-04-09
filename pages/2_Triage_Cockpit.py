"""
Triage Cockpit — Simulated operator decision-making interface.

Three UI variants:
  A: Basic Alert List (Baseline)
  B: AI-Ranked + Evidence (Transparent AI)
  C: Focus Mode (Cognitive Ergonomics)
"""

import time
import streamlit as st
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import UI_EFFECTS
from src.conjunction_data import load_conjunctions
from src.risk_model import predict_risk

st.set_page_config(layout="wide", page_title="Triage Cockpit", page_icon="📡")

# ── State ────────────────────────────────────────────────────────
if "triage_history" not in st.session_state:
    st.session_state.triage_history = []
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# ── Load Data ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_feed():
    try:
        events = load_conjunctions()
        events.sort(key=lambda x: x["max_probability"], reverse=True)
        events = events[:50]
        events = predict_risk(events)
        return events
    except Exception:
        return []

events = get_feed()

if not events:
    st.error("Could not load conjunction data. Run the pipeline first.")
    st.stop()

active_events = [e for e in events if e["event_id"] not in st.session_state.processed_ids]

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    variant = st.radio(
        "UI Variant",
        ["A", "B", "C"],
        format_func=lambda x: f"{x}: {UI_EFFECTS[x]['label']}",
        index=2
    )

    st.caption(
        "**A** — Chronological alert list (baseline)  \n"
        "**B** — AI-ranked with evidence panels  \n"
        "**C** — Focus mode with safety interlocks"
    )

    st.divider()

    elapsed = int(time.time() - st.session_state.start_time)
    mins, secs = divmod(elapsed, 60)
    st.metric("Elapsed", f"{mins:02d}:{secs:02d}")
    st.metric("Processed", len(st.session_state.processed_ids))
    st.metric("Remaining", len(active_events))

    if st.session_state.triage_history:
        st.divider()
        st.caption("Recent actions")
        for entry in reversed(st.session_state.triage_history[-5:]):
            st.text(f"{entry['timestamp']}  {entry['event_id']} → {entry['action']}")

# ── Helpers ──────────────────────────────────────────────────────

def complete_event(event_id, action, true_risk, ai_risk):
    st.session_state.processed_ids.add(event_id)
    st.session_state.triage_history.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "event_id": event_id,
        "action": action,
        "risk": true_risk,
        "ai_risk": ai_risk,
        "variant": variant,
    })
    st.rerun()


def render_card(event, expanded=False):
    risk = event["ai_prediction"]["risk_level"]
    conf = event["ai_prediction"]["confidence"]
    expl = event["ai_prediction"]["explanation"]

    label = f"{event['event_id']}  —  {event['name_1']} vs {event['name_2']}  [{risk.upper()}]"

    with st.expander(label, expanded=expanded):
        # AI insight (Variants B & C)
        if variant in ["B", "C"]:
            st.info(f"**AI Assessment ({conf:.0%} confidence):** {expl['text']}")

        # Telemetry
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TCA", f"{event['time_to_tca_hours']} h")
        c2.metric("Range", f"{event['min_range_km']:.3f} km")
        c3.metric("Prob", f"{event['max_probability']:.2e}")
        c4.metric("Data Age", f"{event['combined_dse']} d")

        # Variant C: SOP guidance
        if variant == "C":
            st.success(f"**SOP:** {expl['sop']}  \n**Next step:** {expl['next_step']}")

        # Actions
        cols = st.columns(4)

        actions = [
            ("execute_maneuver", "Execute Maneuver", risk == "critical"),
            ("prepare_maneuver", "Prepare Plan", risk == "high"),
            ("monitor", "Monitor", risk == "medium"),
            ("log_only", "Dismiss", risk == "low"),
        ]

        for i, (code, lbl, is_primary) in enumerate(actions):
            disabled = variant == "C" and code == "execute_maneuver" and risk != "critical"
            btn_type = "primary" if is_primary else "secondary"

            if cols[i].button(lbl, key=f"{code}_{event['event_id']}", type=btn_type,
                              disabled=disabled, use_container_width=True):
                complete_event(event["event_id"], code, event["risk_level"], risk)


# ── Sorting ──────────────────────────────────────────────────────
display = list(active_events)
if variant in ["B", "C"]:
    display.sort(key=lambda x: x["risk_order"])
else:
    display.sort(key=lambda x: x["time_to_tca_hours"])

# ── Main ─────────────────────────────────────────────────────────
st.title("Triage Cockpit")
st.markdown(f"**Active variant:** {variant} — {UI_EFFECTS[variant]['label']}")

if not active_events:
    st.success("All events processed.")
    if st.button("Reset Session"):
        st.session_state.processed_ids = set()
        st.session_state.triage_history = []
        st.rerun()

elif variant == "C":
    st.subheader("Priority Target")
    render_card(display[0], expanded=True)

    if len(display) > 1:
        st.subheader("Pending Queue")
        for e in display[1:4]:
            render_card(e)
else:
    for e in display[:10]:
        render_card(e, expanded=(variant == "B" and e == display[0]))
