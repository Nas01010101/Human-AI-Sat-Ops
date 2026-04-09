"""
Central configuration for ORBIT-GUARD — Satellite Collision Assessment.

All thresholds, operator profiles, UI variant effects, cost weights,
and output paths for the conjunction-focused pipeline.
"""

import os
import numpy as np

# ─── Reproducibility ───────────────────────────────────────────────
GLOBAL_SEED = 42
RNG = np.random.default_rng(GLOBAL_SEED)

# ─── SOCRATES Data Settings ───────────────────────────────────────
SOCRATES_CSV_URL = "https://celestrak.org/SOCRATES/sort-minRange.csv"
SOCRATES_CACHE_TTL = 6 * 3600  # 6 hours

# ─── Risk-Level Thresholds ────────────────────────────────────────
# Derived from conjunction physics (miss distance + collision probability)
RISK_THRESHOLDS = {
    "critical": {"max_range_km": 1.0,   "min_prob": 1e-4},
    "high":     {"max_range_km": 5.0,   "min_prob": 1e-5},
    "medium":   {"max_range_km": 25.0,  "min_prob": 0},
    # everything else → low
}

RISK_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# ─── Action Mapping ──────────────────────────────────────────────
ACTION_RULES = {
    "critical": "execute_maneuver",   # immediate avoidance burn
    "high":     "prepare_maneuver",   # plan maneuver, notify Flight Director
    "medium":   "monitor",            # track, update with next GP data
    "low":      "log_only",           # record and continue
}

# ─── Feature Names (for ML model) ────────────────────────────────
FEATURE_NAMES = [
    "min_range_km",
    "relative_speed_kms",
    "max_probability",
    "time_to_tca_hours",
    "dse_primary",
    "dse_secondary",
    "combined_dse",
    "kinetic_energy_proxy",
]

# ─── Operator Profile Parameters ──────────────────────────────────
OPERATOR_PROFILES = {
    "novice": {
        "label": "Novice Controller",
        "attention_capacity": 8,
        "base_error_rate": 0.15,
        "automation_reliance": 0.85,
        "verification_propensity": 0.25,
        "checklist_adherence": 0.70,
        "fatigue_growth": 0.0,
        "decision_time_mean": 12.0,  # seconds — conjunction assessment is complex
        "decision_time_std": 4.0,
        "description": "New to conjunction ops, high cognitive load, follows AI blindly",
    },
    "experienced": {
        "label": "Senior Controller",
        "attention_capacity": 15,
        "base_error_rate": 0.05,
        "automation_reliance": 0.50,
        "verification_propensity": 0.55,
        "checklist_adherence": 0.92,
        "fatigue_growth": 0.0,
        "decision_time_mean": 6.0,
        "decision_time_std": 2.0,
        "description": "Pattern recognition, calibrated trust, verifies data freshness",
    },
    "fatigued": {
        "label": "On-Call / Fatigued Controller",
        "attention_capacity": 10,
        "base_error_rate": 0.12,
        "automation_reliance": 0.70,
        "verification_propensity": 0.30,
        "checklist_adherence": 0.55,
        "fatigue_growth": 0.006,  # per-event increase
        "decision_time_mean": 9.0,
        "decision_time_std": 4.5,
        "description": "Slip rate rises with time, checklist skipping under pressure",
    },
}

# ─── UI Variant Effect Modifiers ──────────────────────────────────
UI_EFFECTS = {
    "A": {
        "label": "Basic Alert List",
        "verification_boost": 0.0,
        "reliance_reduction_weak_evidence": 0.0,
        "attention_boost": 0,
        "checklist_required": False,
        "time_cost_evidence": 0.0,
        "time_cost_checklist": 0.0,
    },
    "B": {
        "label": "AI-Ranked + Evidence",
        "verification_boost": 0.25,
        "reliance_reduction_weak_evidence": 0.30,
        "attention_boost": 0,
        "checklist_required": False,
        "time_cost_evidence": 2.5,
        "time_cost_checklist": 0.0,
    },
    "C": {
        "label": "Focus Mode",
        "verification_boost": 0.25,
        "reliance_reduction_weak_evidence": 0.30,
        "attention_boost": 5,
        "checklist_required": True,
        "time_cost_evidence": 2.5,
        "time_cost_checklist": 5.0,
    },
}

# ─── Simulation Settings ──────────────────────────────────────────
N_SIMULATION_RUNS = 500

# ─── Operational Cost Weights ─────────────────────────────────────
COST_WEIGHTS = {
    "missed_critical": 50,     # missed critical conjunction
    "missed_high": 20,         # missed high-risk conjunction
    "false_maneuver": 8,       # unnecessary maneuver (fuel cost)
    "false_prepare": 3,        # unnecessary preparation (time cost)
    "correct": 0,
}

# ─── Output Paths ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
