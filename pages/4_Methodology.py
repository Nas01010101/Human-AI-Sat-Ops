"""
Methodology — Data sources, model architecture, simulation design, and limitations.
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RISK_THRESHOLDS, FEATURE_NAMES, OPERATOR_PROFILES,
    UI_EFFECTS, COST_WEIGHTS, N_SIMULATION_RUNS
)

st.set_page_config(page_title="Methodology", page_icon="📄", layout="wide")

st.title("Methodology")
st.caption("Technical documentation of the ORBIT-GUARD research platform.")

# ═══════════════════════════════════════════════════════════════════
# 1. DATA SOURCE
# ═══════════════════════════════════════════════════════════════════
st.header("1. Data Source")

st.markdown("""
All conjunction data is sourced from 
**[CelesTrak SOCRATES](https://celestrak.org/SOCRATES/)** 
(Satellite Orbital Conjunction Reports Assessing Threatening Encounters in Space).

SOCRATES provides daily conjunction screening reports for the entire tracked 
catalog of Earth-orbiting objects. Each record represents a predicted close 
approach between two cataloged objects.
""")

st.subheader("Fields Used")

st.markdown("""
| Field | Description |
|-------|-------------|
| `NORAD_CAT_ID_1/2` | Catalog identifiers for primary and secondary objects |
| `OBJECT_NAME_1/2` | Human-readable names (e.g., Starlink-5382) |
| `TCA` | Time of Closest Approach (UTC) |
| `MIN_RNG` | Minimum range at closest approach (km) |
| `MAX_PROB` | Maximum collision probability (dimensionless) |
| `RELATIVE_SPEED` | Relative velocity at TCA (km/s) |
| `DSE_1/2` | Days Since Epoch — age of the tracking data for each object |
""")

st.markdown("""
**Update frequency:** The SOCRATES database is updated approximately every 8 hours.  
**Coverage:** All tracked objects in the USSPACECOM catalog (~47,000 objects).  
**Citation:** T.S. Kelso, CelesTrak, https://celestrak.org/SOCRATES/
""")

# ═══════════════════════════════════════════════════════════════════
# 2. DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════
st.header("2. Data Processing Pipeline")

st.markdown("""
Raw SOCRATES CSV records are processed through the following steps:
""")

st.subheader("2.1 Feature Engineering")

st.markdown(f"""
Each conjunction record is enriched with derived features:

| Feature | Computation |
|---------|-------------|
| `min_range_km` | Directly from `MIN_RNG` |
| `max_probability` | Directly from `MAX_PROB` |
| `relative_speed_kms` | Directly from `RELATIVE_SPEED` |
| `time_to_tca_hours` | `TCA - current_time`, in hours |
| `dse_primary` | Days Since Epoch for the primary object |
| `dse_secondary` | Days Since Epoch for the secondary object |
| `combined_dse` | `max(dse_primary, dse_secondary)` |
| `kinetic_energy_proxy` | `relative_speed² × estimated_mass` — rough indicator of collision severity |

A total of **{len(FEATURE_NAMES)} features** are used for classification:  
`{', '.join(FEATURE_NAMES)}`
""")

st.subheader("2.2 Risk Labeling")

st.markdown("""
Ground-truth risk labels are assigned using physics-based thresholds, 
**not** human annotation. This is a key design choice — the AI model learns 
to approximate deterministic rules, allowing us to isolate the effect of 
UI design on operator decisions rather than model accuracy.
""")

st.markdown("**Threshold rules:**")
for level, thresholds in RISK_THRESHOLDS.items():
    st.markdown(
        f"- **{level.upper()}**: Range ≤ {thresholds['max_range_km']} km "
        f"AND Probability ≥ {thresholds['min_prob']:.0e}"
    )
st.markdown("- **LOW**: Everything else")

st.subheader("2.3 Performance Optimization")

st.markdown("""
The full SOCRATES catalog contains ~114,000 conjunction records per day. 
For practical reasons, the pipeline:

1. **Trains** the ML model on the full dataset (~114k events)
2. **Filters** to the top 2,000 highest-risk events for prediction and UI display
3. **Samples** 500 events for the Monte Carlo operator simulation

This keeps pipeline runtime under 60 seconds while preserving the most 
operationally relevant events.
""")

# ═══════════════════════════════════════════════════════════════════
# 3. ML MODEL
# ═══════════════════════════════════════════════════════════════════
st.header("3. Machine Learning Model")

st.markdown("""
**Algorithm:** Random Forest Classifier (scikit-learn)  
**Parameters:** 100 trees, max depth 5, seed 42  
**Validation:** 5-fold stratified cross-validation  
**Training accuracy:** >99% (expected, since labels are deterministic functions of features)

The model's purpose is not to discover hidden patterns, but to provide a 
**plausible AI component** that the operator interacts with. The high accuracy 
is by design — it lets us study trust calibration and UI effects without 
confounding from poor model performance.
""")

st.subheader("Explanation Generation")

st.markdown("""
For each prediction, the system generates a structured explanation:

1. **Risk factors** — identifies which features drove the prediction 
   (e.g., "Close approach distance (0.12 km), High collision probability (3.2e-4)")
2. **SOP reference** — maps to a Standard Operating Procedure code
3. **Next step** — actionable guidance (e.g., "Verify ephemeris and authorize maneuver burn")

Explanations use a simplified feature-importance approach based on 
global importance × local deviation from the mean. This is not 
SHAP or LIME — it is an approximation suitable for this prototype.
""")

# ═══════════════════════════════════════════════════════════════════
# 4. SIMULATION DESIGN
# ═══════════════════════════════════════════════════════════════════
st.header("4. Operator Simulation Design")

st.subheader("4.1 Operator Profiles")

st.markdown("Three operator profiles model different expertise levels:")

for name, profile in OPERATOR_PROFILES.items():
    with st.expander(f"{profile['label']} ({name})"):
        st.markdown(f"*{profile['description']}*")
        cols = st.columns(3)
        cols[0].metric("Base Error Rate", f"{profile['base_error_rate']:.0%}")
        cols[1].metric("Automation Reliance", f"{profile['automation_reliance']:.0%}")
        cols[2].metric("Verification Rate", f"{profile['verification_propensity']:.0%}")

st.subheader("4.2 UI Variants")

st.markdown("Three interface designs are compared:")

for key, ui in UI_EFFECTS.items():
    st.markdown(
        f"- **Variant {key} ({ui['label']})**: "
        f"Verification boost = {ui['verification_boost']:.0%}, "
        f"Checklist required = {ui['checklist_required']}"
    )

st.subheader("4.3 Monte Carlo Procedure")

st.markdown(f"""
The simulation runs **{N_SIMULATION_RUNS} independent trials** for each 
combination of operator profile × UI variant (3 × 3 = 9 conditions).

Each trial:
1. Presents the sampled events to the simulated operator
2. The operator model decides an action based on AI prediction, 
   profile parameters, and UI variant effects
3. Decisions are compared against ground truth to compute accuracy, 
   miss rate, and weighted cost
""")

st.subheader("4.4 Cost Function")

st.markdown("Outcomes are weighted to reflect operational severity:")

for outcome, weight in COST_WEIGHTS.items():
    st.markdown(f"- **{outcome.replace('_', ' ').title()}**: ×{weight}")

# ═══════════════════════════════════════════════════════════════════
# 5. RESULTS INTERPRETATION
# ═══════════════════════════════════════════════════════════════════
st.header("5. Interpreting Results")

st.markdown("""
The Simulation Results page shows three metrics:

- **Decision Accuracy**: Proportion of events where the operator's action 
  matched the ground-truth optimal response
- **Critical Miss Rate**: Proportion of critical/high events that were 
  incorrectly dismissed or under-triaged
- **Operational Cost**: Weighted sum of error penalties — lower is better

Key comparisons:
- **Variant A vs B**: Does adding AI evidence improve accuracy?
- **Variant B vs C**: Does adding cognitive aids (focus mode, interlocks) 
  further reduce errors?
- **Novice vs Experienced**: How much does expertise compensate for a 
  weaker interface?
- **Fatigued profile**: Does Variant C's structure protect against 
  fatigue-induced errors?
""")

# ═══════════════════════════════════════════════════════════════════
# 6. LIMITATIONS
# ═══════════════════════════════════════════════════════════════════
st.header("6. Limitations")

st.markdown("""
This is an undergraduate research prototype. The following limitations 
should be considered when interpreting results:

1. **Synthetic ground truth.** Risk labels are derived from deterministic 
   physics thresholds, not from expert human annotation or validated 
   conjunction assessment databases.

2. **Simulated operators.** No real human subjects were tested. The operator 
   models are parameterized approximations of cognitive processes, not 
   validated against empirical data.

3. **Simplified orbital mechanics.** The 3D encounter view uses schematic 
   geometry, not full SGP4/SDP4 orbital propagation. Actual conjunction 
   assessment requires covariance-based screening.

4. **Model interpretability.** The explanation system uses global feature 
   importance, not instance-level attribution methods (SHAP, LIME). 
   Explanations may not reflect the true local decision boundary.

5. **No temporal dynamics.** Events are treated independently. In practice, 
   conjunction assessments evolve over multiple screening epochs with 
   updated tracking data.

6. **Cost function calibration.** The penalty weights (missed critical = 50×, 
   false alarm = 8×) are notional estimates, not derived from operational 
   cost data.

7. **Single data source.** Only the SOCRATES minimum-range sort is used. 
   Operational conjunction assessment integrates multiple data sources 
   (owner/operator ephemerides, high-accuracy sensors).
""")

st.divider()
st.caption(
    "For questions about this project, please refer to the repository README. "
    "Data provided by CelesTrak (T.S. Kelso). Not for operational use."
)
