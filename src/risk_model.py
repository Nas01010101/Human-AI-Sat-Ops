"""
AI Risk Model — Random Forest classifier for collision risk assessment.

Trains on SOCRATES conjunction features to predict risk level (critical/high/medium/low).
Generates inline explanations (feature importance) and confidence scores.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import (
    FEATURE_NAMES, RISK_ORDER, MODELS_DIR, GLOBAL_SEED
)

# ─── Model Paths ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODELS_DIR, "risk_rf_model.joblib")


def flatten_features(events: List[Dict]) -> pd.DataFrame:
    """Convert list of event dicts to feature DataFrame."""
    rows = []
    for e in events:
        row = dict(e["features"])
        row["risk_level"] = e["risk_level"]
        row["event_id"] = e["event_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def train_risk_model(events: List[Dict]) -> Dict:
    """
    Train Random Forest to predict risk level from conjunction features.

    Uses shallow depth to intentionally limit accuracy — a real-world AI
    system won't perfectly replicate physics rules, especially near decision
    boundaries. This makes the human-in-the-loop comparison meaningful.
    """
    df = flatten_features(events)
    X = df[FEATURE_NAMES]
    y = df["risk_level"]

    # Shallow tree = realistic ~85% accuracy (boundary cases are hard)
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,       # intentionally limited — mimics real AI uncertainty
        random_state=GLOBAL_SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)

    if len(y.unique()) < 2 or len(y) < 10:
        rf.fit(X, y)
        acc = accuracy_score(y, rf.predict(X))
    else:
        y_pred = cross_val_predict(rf, X, y, cv=cv)
        acc = accuracy_score(y, y_pred)
        rf.fit(X, y)

    joblib.dump(rf, MODEL_PATH)
    importances = dict(zip(FEATURE_NAMES, rf.feature_importances_))

    return {
        "accuracy": acc,
        "feature_importances": importances,
        "model_path": MODEL_PATH,
        "n_samples": len(df)
    }


def predict_risk(events: List[Dict]) -> List[Dict]:
    """
    Generate AI predictions, confidence scores, and explanations for events.

    Confidence is calibrated to reflect genuine uncertainty near decision
    boundaries (e.g. miss distance just above/below threshold), not just
    raw RF vote fractions. This produces realistic AI behavior where the
    model is confident on clear cases and uncertain on edge cases.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")

    rf = joblib.load(MODEL_PATH)
    df = flatten_features(events)
    X = df[FEATURE_NAMES]

    probs = rf.predict_proba(X)
    classes = rf.classes_
    importances = rf.feature_importances_
    means = X.mean()
    stds = X.std().replace(0, 1)

    # Boundary proximity: how close is each event to a risk threshold?
    # Closer to boundary → lower confidence, higher chance of wrong prediction
    RANGE_BOUNDARIES = [1.0, 5.0, 25.0]   # critical/high/medium thresholds
    PROB_BOUNDARIES  = [1e-4, 1e-5]

    results = []
    for i, event in enumerate(events):
        top_idx = np.argmax(probs[i])
        pred_label = classes[top_idx]
        raw_conf = float(np.max(probs[i]))

        # Compute boundary proximity penalty
        r = event["features"]["min_range_km"]
        p = event["features"]["max_probability"]

        range_penalty = min(
            abs(r - b) / (b + 1e-9) for b in RANGE_BOUNDARIES
        )
        prob_penalty = min(
            abs(np.log10(max(p, 1e-12)) - np.log10(b)) for b in PROB_BOUNDARIES
        )

        # Normalize penalties: close to boundary (penalty < 0.2) → reduce confidence
        boundary_factor = min(range_penalty, 1.0) * min(prob_penalty / 2.0, 1.0)
        calibrated_conf = raw_conf * (0.6 + 0.4 * min(boundary_factor, 1.0))
        calibrated_conf = float(np.clip(calibrated_conf, 0.3, 0.99))

        # Evidence strength from data staleness (used by operator model)
        evidence_strength = event.get("evidence_strength", "moderate")

        explanation = _generate_explanation(
            event, pred_label, calibrated_conf, X.iloc[i], means, stds, importances
        )

        event["ai_prediction"] = {
            "risk_level": pred_label,
            "confidence": round(calibrated_conf, 3),
            "raw_confidence": round(raw_conf, 3),
            "evidence_strength": evidence_strength,
            "explanation": explanation,
            "action_recommendation": _recommend_action(pred_label, event),
        }
        results.append(event)

    return results


def _recommend_action(risk_level: str, event: Dict) -> str:
    """Map predicted risk + context to action."""
    # AI logic matches standard SOPs
    if risk_level == "critical":
        return "execute_maneuver"
    elif risk_level == "high":
        return "prepare_maneuver"
    elif risk_level == "medium":
        # Check for degrading trend or stale data
        if event["features"]["combined_dse"] > 1.5:
             # Stale data → escalate medium to prepare
             return "prepare_maneuver"
        return "monitor"
    else:
        return "log_only"


def _generate_explanation(event: Dict, pred_label: str, confidence: float,
                          row: pd.Series, means: pd.Series, stds: pd.Series, 
                          importances: np.ndarray) -> Dict:
    """
    Construct a structured explanation for the operator.
    """
    # 1. Identify driving factors (high value * high importance)
    z_scores = (row - means) / stds
    # We care about directionality relative to risk
    # tailored logic: low range = high risk, high speed = high risk
    
    factors = []
    
    # Min Range (lower is riskier)
    if row["min_range_km"] < means["min_range_km"] and row["min_range_km"] < 5.0:
        factors.append(f"Close approach distance ({row['min_range_km']:.2f} km)")
    
    # Probability (higher is riskier)
    if row["max_probability"] > 1e-5:
        factors.append(f"High collision probability ({row['max_probability']:.1e})")
        
    # Data Staleness (high is riskier/uncertain)
    if row["combined_dse"] > 2.0:
        factors.append(f"Stale telemetry ({row['combined_dse']:.1f} days old)")

    # Kinetic Energy (high is riskier)
    if row["kinetic_energy_proxy"] > means["kinetic_energy_proxy"]:
         factors.append(f"High relative velocity ({row['relative_speed_kms']:.1f} km/s)")
         
    # Text summary
    if not factors:
        factors.append("All values within nominal thresholds")
        
    text = (
        f"AI assesses {pred_label.upper()} risk ({confidence*100:.0f}% confidence). "
        f"Key drivers: {', '.join(factors[:2])}."
    )
    
    # SOP Guidance
    sop_map = {
        "critical": "SOP-909: COLLISION AVOIDANCE MANEUVER",
        "high": "SOP-800: PREPARE MANEUVER PLAN",
        "medium": "SOP-100: INCREASED MONITORING",
        "low": "SOP-010: ROUTINE LOGGING",
    }
    
    next_step_map = {
        "critical": "Verify ephemeris and authorize maneuver burn.",
        "high": "Task FDS for higher-fidelity screening.",
        "medium": "Request updated tracking data (OD).",
        "low": "No action required.",
    }

    return {
        "text": text,
        "factors": factors,
        "sop": sop_map.get(pred_label, "SOP-000"),
        "next_step": next_step_map.get(pred_label, "Monitor situation."),
    }


if __name__ == "__main__":
    from src.conjunction_data import load_conjunctions
    
    print("Training risk model...")
    events = load_conjunctions()
    metrics = train_risk_model(events)
    print(f"  Training Accuracy: {metrics['accuracy']:.3f}")
    
    print("\nPredicting on sample...")
    events_pred = predict_risk(events[:5])
    for e in events_pred:
        p = e["ai_prediction"]
        print(f"  {e['event_id']} -> {p['risk_level']} ({p['confidence']:.2f})")
        print(f"    Ex: {p['explanation']['text']}")
