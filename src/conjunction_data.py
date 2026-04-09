"""
Conjunction data module — fetches real satellite close-approach data
from CelesTrak SOCRATES.

Data source:
  CelesTrak SOCRATES-Plus CSV
  https://celestrak.org/SOCRATES/sort-minRange.csv
  Fields: NORAD_CAT_ID_1, OBJECT_NAME_1, DSE_1, NORAD_CAT_ID_2,
          OBJECT_NAME_2, DSE_2, TCA, TCA_RANGE, TCA_RELATIVE_SPEED,
          MAX_PROB, DILUTION

All data is public domain. Cached locally to avoid repeated API calls.
"""

import csv
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from src.config import (
    SOCRATES_CSV_URL, SOCRATES_CACHE_TTL,
    RISK_THRESHOLDS, ACTION_RULES, RISK_ORDER,
    FEATURE_NAMES, CACHE_DIR, GLOBAL_SEED,
)

# ─── Cache ────────────────────────────────────────────────────────
_CACHE_FILE = os.path.join(CACHE_DIR, "socrates_latest.csv")
_CACHE_JSON = os.path.join(CACHE_DIR, "conjunctions_enriched.json")


def _is_cache_valid() -> bool:
    """Check if the CSV cache exists and is fresh."""
    if not os.path.exists(_CACHE_FILE):
        return False
    age = time.time() - os.path.getmtime(_CACHE_FILE)
    return age < SOCRATES_CACHE_TTL


# ─── Fetch ────────────────────────────────────────────────────────

def fetch_socrates_csv(force: bool = False) -> str:
    """
    Fetch the latest SOCRATES CSV from CelesTrak.

    Returns raw CSV text. Uses local cache if still valid.
    Falls back to bundled sample data if network is unavailable.
    """
    if not force and _is_cache_valid():
        with open(_CACHE_FILE, "r") as f:
            return f.read()

    if not HAS_REQUESTS:
        return _load_fallback()

    try:
        resp = requests.get(SOCRATES_CSV_URL, timeout=30)
        resp.raise_for_status()
        text = resp.text

        # Basic validation: check for header or content length
        if len(text) < 100 or "NORAD" not in text:
            print(f"  [conjunction_data] Warning: Downloaded CSV seems invalid/empty. Using cache/fallback.")
            raise ValueError("Invalid CSV content")

        # Save to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            f.write(text)

        return text
    except (requests.RequestException, ValueError) as e:
        print(f"  [conjunction_data] Warning: fetch failed: {e}")
        # Try stale cache
        if os.path.exists(_CACHE_FILE):
            print("  [conjunction_data] Loading stale cache.")
            with open(_CACHE_FILE, "r") as f:
                return f.read()
        print("  [conjunction_data] Cache unavailable. Using synthetic fallback.")
        return _load_fallback()


def _load_fallback() -> str:
    """Load bundled sample data for offline use."""
    sample_path = os.path.join(CACHE_DIR, "socrates_sample.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            return f.read()
    # Generate minimal synthetic fallback
    return _generate_synthetic_csv()


def _generate_synthetic_csv(n: int = 200) -> str:
    """Generate synthetic SOCRATES-like CSV for offline/demo use."""
    rng = np.random.default_rng(GLOBAL_SEED)
    header = "NORAD_CAT_ID_1,OBJECT_NAME_1,DSE_1,NORAD_CAT_ID_2,OBJECT_NAME_2,DSE_2,TCA,TCA_RANGE,TCA_RELATIVE_SPEED,MAX_PROB,DILUTION\n"
    rows = []

    sat_names = [
        "STARLINK-1007", "STARLINK-1052", "STARLINK-2131", "COSMOS 2251 DEB",
        "FENGYUN 1C DEB", "IRIDIUM 33 DEB", "CZ-6A DEB", "SL-16 R/B",
        "NOAA 17", "ISS (ZARYA)", "COSMOS 1408 DEB", "GLOBALSTAR M069",
        "ONEWEB-0012", "FLOCK 4P-1", "LEMUR-2-ALEX", "SPACEBEE-88",
        "DOVE-3", "SKYSAT-C12", "YAOGAN-35A", "SENTINEL-6A",
    ]
    cat_ids = list(range(25544, 25544 + len(sat_names)))

    for i in range(n):
        idx1, idx2 = rng.choice(len(sat_names), size=2, replace=False)
        # Exponential distribution biased toward close approaches
        min_range = float(rng.exponential(8.0) + 0.01)
        rel_speed = float(rng.uniform(0.1, 15.5))
        # Probability inversely related to range
        max_prob = float(np.clip(1e-2 / (min_range ** 2 + 0.01), 1e-12, 1.0))
        dse1 = round(float(rng.uniform(0.05, 3.0)), 2)
        dse2 = round(float(rng.uniform(0.05, 3.0)), 2)
        dilution = round(float(rng.uniform(0.1, 50.0)), 3)
        # TCA in the next 7 days
        tca_dt = datetime(2025, 2, 18, 0, 0, 0, tzinfo=timezone.utc)
        tca_offset_sec = int(rng.uniform(0, 7 * 86400))
        tca = datetime.fromtimestamp(
            tca_dt.timestamp() + tca_offset_sec, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        rows.append(
            f"{cat_ids[idx1]},{sat_names[idx1]} [+],{dse1},"
            f"{cat_ids[idx2]},{sat_names[idx2]} [+],{dse2},"
            f"{tca},{min_range:.3f},{rel_speed:.3f},{max_prob:.6e},{dilution:.3f}"
        )

    csv_text = header + "\n".join(rows)
    # Save as sample
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "socrates_sample.csv"), "w") as f:
        f.write(csv_text)
    return csv_text


# ─── Parse ────────────────────────────────────────────────────────

def parse_socrates_csv(csv_text: str) -> List[Dict]:
    """
    Parse SOCRATES CSV into a list of conjunction records.

    Each record is a dict with raw SOCRATES fields plus computed features.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    records = []

    for row in reader:
        try:
            rec = {
                "norad_id_1": int(row["NORAD_CAT_ID_1"]),
                "name_1": row["OBJECT_NAME_1"].strip(),
                "dse_1": float(row["DSE_1"]),
                "norad_id_2": int(row["NORAD_CAT_ID_2"]),
                "name_2": row["OBJECT_NAME_2"].strip(),
                "dse_2": float(row["DSE_2"]),
                "tca": row["TCA"].strip(),
                "min_range_km": float(row["TCA_RANGE"]),
                "relative_speed_kms": float(row["TCA_RELATIVE_SPEED"]),
                "max_probability": float(row["MAX_PROB"]),
                "dilution_km": float(row["DILUTION"]),
            }
            records.append(rec)
        except (ValueError, KeyError):
            continue

    return records


# ─── Enrich ───────────────────────────────────────────────────────

def _compute_time_to_tca(tca_str: str) -> float:
    """Compute hours from now to TCA. Returns 0 if TCA is in the past."""
    try:
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
            try:
                tca_dt = datetime.strptime(tca_str, fmt).replace(tzinfo=timezone.utc)
                delta = (tca_dt - datetime.now(timezone.utc)).total_seconds() / 3600
                return max(delta, 0.0)
            except ValueError:
                continue
        return 24.0  # default if parse fails
    except Exception:
        return 24.0


def assign_risk_level(min_range_km: float, max_prob: float) -> str:
    """Assign risk level from physics-based thresholds."""
    if min_range_km < RISK_THRESHOLDS["critical"]["max_range_km"] and \
       max_prob > RISK_THRESHOLDS["critical"]["min_prob"]:
        return "critical"
    if min_range_km < RISK_THRESHOLDS["high"]["max_range_km"] or \
       max_prob > RISK_THRESHOLDS["high"]["min_prob"]:
        return "high"
    if min_range_km < RISK_THRESHOLDS["medium"]["max_range_km"]:
        return "medium"
    return "low"


def enrich_conjunctions(records: List[Dict]) -> List[Dict]:
    """
    Add computed features, risk labels, and actions to raw records.

    Features:
      - time_to_tca_hours: hours until closest approach
      - combined_dse: max data staleness (worst case)
      - kinetic_energy_proxy: relative_speed² (destructive potential)
      - risk_level: critical/high/medium/low
      - action: execute_maneuver / prepare_maneuver / monitor / log_only
      - evidence_strength: strong/moderate/weak
    """
    enriched = []
    for i, rec in enumerate(records):
        event = dict(rec)

        # Computed features
        event["time_to_tca_hours"] = round(_compute_time_to_tca(rec["tca"]), 2)
        event["combined_dse"] = round(max(rec["dse_1"], rec["dse_2"]), 2)
        event["kinetic_energy_proxy"] = round(rec["relative_speed_kms"] ** 2, 2)

        # Risk classification
        risk = assign_risk_level(rec["min_range_km"], rec["max_probability"])
        event["risk_level"] = risk
        event["action_true"] = ACTION_RULES[risk]
        event["risk_order"] = RISK_ORDER[risk]

        # Evidence strength (based on data freshness)
        combined_dse = event["combined_dse"]
        if combined_dse < 0.5:
            event["evidence_strength"] = "strong"
        elif combined_dse < 1.5:
            event["evidence_strength"] = "moderate"
        else:
            event["evidence_strength"] = "weak"

        # Event ID
        event["event_id"] = f"CNJ-{i+1:04d}"

        # Text summary for the UI
        risk_tag = {
            "critical": "🔴 CRITICAL",
            "high": "🟠 HIGH",
            "medium": "🟡 MEDIUM",
            "low": "🟢 LOW",
        }[risk]

        event["text"] = (
            f"{risk_tag} | {rec['name_1']} ↔ {rec['name_2']} — "
            f"miss {rec['min_range_km']:.3f} km, "
            f"TCA {rec['tca']}, "
            f"Pc={rec['max_probability']:.2e}"
        )

        # Evidence lines (CDM-style)
        event["evidence_lines"] = [
            f"[CDM] Primary: {rec['name_1']} (NORAD {rec['norad_id_1']})",
            f"[CDM] Secondary: {rec['name_2']} (NORAD {rec['norad_id_2']})",
            f"[CDM] TCA: {rec['tca']} | Min range: {rec['min_range_km']:.3f} km",
            f"[CDM] Relative velocity: {rec['relative_speed_kms']:.3f} km/s",
            f"[CDM] Max collision probability: {rec['max_probability']:.2e}",
            f"[CDM] Data staleness: primary {rec['dse_1']:.1f}d, secondary {rec['dse_2']:.1f}d",
            f"[CDM] Dilution threshold: {rec['dilution_km']:.3f} km",
        ]

        # Feature vector for ML
        event["features"] = {
            "min_range_km": rec["min_range_km"],
            "relative_speed_kms": rec["relative_speed_kms"],
            "max_probability": rec["max_probability"],
            "time_to_tca_hours": event["time_to_tca_hours"],
            "dse_primary": rec["dse_1"],
            "dse_secondary": rec["dse_2"],
            "combined_dse": event["combined_dse"],
            "kinetic_energy_proxy": event["kinetic_energy_proxy"],
        }

        enriched.append(event)

    return enriched


# ─── Full Pipeline ────────────────────────────────────────────────

def load_conjunctions(force_refresh: bool = False) -> List[Dict]:
    """
    Full pipeline: fetch → parse → enrich.

    Returns a list of enriched conjunction event dicts.
    """
    csv_text = fetch_socrates_csv(force=force_refresh)
    records = parse_socrates_csv(csv_text)
    enriched = enrich_conjunctions(records)
    return enriched


def export_for_web(events: List[Dict], filename: str = "conjunctions.json") -> str:
    """Export enriched events as JSON for the Streamlit UI."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    # Make JSON-serializable
    clean = []
    for e in events:
        c = dict(e)
        # Convert any numpy types
        for k, v in c.items():
            if isinstance(v, (np.integer,)):
                c[k] = int(v)
            elif isinstance(v, (np.floating,)):
                c[k] = float(v)
            if isinstance(v, dict):
                c[k] = {kk: float(vv) if isinstance(vv, (np.floating,)) else vv
                         for kk, vv in v.items()}
        clean.append(c)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    return path


# ─── CLI Test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching SOCRATES conjunction data...")
    events = load_conjunctions()
    print(f"  Loaded {len(events)} conjunctions")

    # Risk distribution
    from collections import Counter
    risks = Counter(e["risk_level"] for e in events)
    actions = Counter(e["action_true"] for e in events)
    print(f"  Risk levels: {dict(risks)}")
    print(f"  Actions: {dict(actions)}")

    # Top 5 closest
    by_range = sorted(events, key=lambda e: e["min_range_km"])
    print("\n  Top 5 closest approaches:")
    for e in by_range[:5]:
        print(f"    {e['event_id']}: {e['name_1']} ↔ {e['name_2']} — "
              f"{e['min_range_km']:.3f} km, Pc={e['max_probability']:.2e}, "
              f"risk={e['risk_level']}")

    # Export
    path = export_for_web(events)
    print(f"\n  Exported to {path}")
