"""
Operator Simulation Module — Models human decision making in collision scenarios.

Simulates how operators (Novice, Experienced, Fatigued) interact with
different UI variants (A=Basic, B=Evidence, C=Focus) when assessing
AI risk predictions.

Key components:
  1. OperatorModel: State machine for attention, trust, verification
  2. run_simulation: Monte Carlo loop
  3. Metrics: Accuracy, Miss Rate, Cost, False Alarm Rate
"""

import math
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.config import (
    OPERATOR_PROFILES, UI_EFFECTS, COST_WEIGHTS,
    N_SIMULATION_RUNS, RNG
)


class OperatorModel:
    """
    Simulates a human operator's cognitive process during a shift.

    Improvements over baseline:
    - Dynamic trust: trust_in_ai updates based on observed AI correctness
    - Confidence-aware verification: low AI confidence triggers more scrutiny
    - Evidence quality modulation: Variant B/C evidence_strength affects decision
    - Fatigue compounds error rate per-event (not just a flat offset)
    """

    def __init__(self, profile_name: str, ui_variant: str):
        self.profile = OPERATOR_PROFILES[profile_name]
        self.ui = UI_EFFECTS[ui_variant]
        self.variant_code = ui_variant

        self.fatigue_level = 0.0
        self.trust_in_ai = self.profile["automation_reliance"]
        self._recent_ai_outcomes = []   # track last N AI decisions for trust update

    def process_event(self, event: Dict) -> Dict:
        ai_pred = event.get("ai_prediction", {})
        ai_risk = ai_pred.get("risk_level", "low")
        ai_conf = ai_pred.get("confidence", 0.5)
        evidence_strength = ai_pred.get("evidence_strength", "moderate")
        true_risk = event["risk_level"]

        # 1. Fatigue accumulates per event
        effective_error_rate = min(
            self.profile["base_error_rate"] + self.fatigue_level, 0.6
        )
        self.fatigue_level += self.profile.get("fatigue_growth", 0.0)

        # 2. Verification probability
        #    Base: profile propensity + UI boost
        #    Increase if: AI confidence is low, evidence is weak
        #    Decrease if: trust is high AND AI is confident (complacency)
        p_verify = self.profile["verification_propensity"] + self.ui["verification_boost"]

        # Low AI confidence → operator more likely to double-check
        if ai_conf < 0.6:
            p_verify = min(p_verify + 0.25, 1.0)
        elif ai_conf > 0.9 and self.trust_in_ai > 0.75:
            p_verify *= 0.55   # complacency effect

        # Variant B/C: weak evidence explicitly shown → triggers verification
        if self.variant_code in ["B", "C"] and evidence_strength == "weak":
            p_verify = min(p_verify + 0.20, 1.0)

        is_verified = RNG.random() < p_verify

        # 3. Decision
        if is_verified:
            perceived_risk = true_risk if RNG.random() > effective_error_rate else ai_risk
        else:
            perceived_risk = ai_risk

        # 4. Checklist interlock (Variant C)
        from src.config import ACTION_RULES
        chosen_action = ACTION_RULES.get(perceived_risk, "log_only")

        time_taken = self.profile["decision_time_mean"]
        if is_verified:
            time_taken += self.ui["time_cost_evidence"]
        if self.ui["checklist_required"] and perceived_risk in ["critical", "high"]:
            time_taken += self.ui["time_cost_checklist"]

        # 5. Dynamic trust update
        #    Operator can only observe AI correctness on events they verified
        if is_verified:
            ai_was_correct = (ai_risk == true_risk)
            self._recent_ai_outcomes.append(ai_was_correct)
            if len(self._recent_ai_outcomes) > 10:
                self._recent_ai_outcomes.pop(0)
            # Bayesian-style trust update: drift toward observed accuracy
            observed_acc = sum(self._recent_ai_outcomes) / len(self._recent_ai_outcomes)
            self.trust_in_ai += 0.05 * (observed_acc - self.trust_in_ai)
            self.trust_in_ai = float(np.clip(self.trust_in_ai, 0.1, 0.99))

        return {
            "event_id": event["event_id"],
            "true_risk": true_risk,
            "ai_risk": ai_risk,
            "operator_risk": perceived_risk,
            "action_taken": chosen_action,
            "action_correct": chosen_action == event["action_true"],
            "verification_performed": is_verified,
            "time_taken": time_taken,
            "trust_at_decision": round(self.trust_in_ai, 3),
        }


def run_simulation(events: List[Dict], n_runs: int = N_SIMULATION_RUNS) -> List[Dict]:
    """
    Run Monte Carlo simulation across all profiles × variants.
    """
    from src.risk_model import predict_risk
    
    # Pre-calculate AI predictions once
    events_with_ai = predict_risk(events)
    
    results = []
    
    profiles = ["novice", "experienced", "fatigued"]
    variants = ["A", "B", "C"]
    
    total_steps = len(profiles) * len(variants) * n_runs
    step = 0
    
    print(f"Starting simulation: {len(events)} events x {n_runs} runs x {len(profiles)} profiles x {len(variants)} variants")
    
    for prof in profiles:
        for var in variants:
            # Metrics accumulators
            correct_counts = []
            miss_counts = []
            false_alarms = []
            costs = []
            
            for _ in range(n_runs):
                op = OperatorModel(prof, var)
                shift_cost = 0
                shift_correct = 0
                shift_miss = 0
                shift_fa = 0
                
                for event in events_with_ai:
                    res = op.process_event(event)
                    
                    if res["action_correct"]:
                        shift_correct += 1
                    
                    # Calculate Cost
                    c = 0
                    true_risk = res["true_risk"]
                    action = res["action_taken"]
                    if true_risk == "critical" and action != "execute_maneuver":
                        c = COST_WEIGHTS["missed_critical"]
                        shift_miss += 1
                    elif true_risk == "high" and action not in ["execute_maneuver", "prepare_maneuver"]:
                        c = COST_WEIGHTS["missed_high"]
                        shift_miss += 1
                    elif action == "execute_maneuver" and true_risk not in ["critical", "high"]:
                        c = COST_WEIGHTS["false_maneuver"]
                        shift_fa += 1
                    elif action == "prepare_maneuver" and true_risk in ["medium", "low"]:
                        c = COST_WEIGHTS["false_prepare"]
                        shift_fa += 1
                    
                    shift_cost += c
                
                correct_counts.append(shift_correct)
                miss_counts.append(shift_miss)
                false_alarms.append(shift_fa)
                costs.append(shift_cost)
            
            # Aggregate stats for this condition
            n_ev = len(events)
            results.append({
                "profile": prof,
                "variant": var,
                "accuracy_mean": np.mean(correct_counts) / n_ev,
                "accuracy_std": np.std(correct_counts) / n_ev,
                "miss_rate_mean": np.mean(miss_counts) / n_ev,
                "false_alarm_mean": np.mean(false_alarms) / n_ev,
                "cost_mean": np.mean(costs),
                "cost_std": np.std(costs)
            })
            
            step += n_runs
            
    return results


if __name__ == "__main__":
    from src.conjunction_data import load_conjunctions
    
    events = load_conjunctions()
    # Run a small test sim
    res = run_simulation(events[:20], n_runs=10)
    
    from pprint import pprint
    pprint(res)
