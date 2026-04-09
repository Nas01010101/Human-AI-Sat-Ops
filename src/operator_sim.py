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
    Based on Wickens' SEEV mode (Salience, Effort, Expectancy, Value)
    and trusting automation literature (Parasuraman & Riley).
    """

    def __init__(self, profile_name: str, ui_variant: str):
        self.profile = OPERATOR_PROFILES[profile_name]
        self.ui = UI_EFFECTS[ui_variant]
        self.variant_code = ui_variant
        
        # State
        self.fatigue_level = 0.0          # Increases over shift
        self.current_attention = self.profile["attention_capacity"] + self.ui["attention_boost"]
        self.trust_in_ai = self.profile["automation_reliance"]

    def process_event(self, event: Dict) -> Dict:
        """
        Operator decides on an action for a single collision event.
        
        Steps:
          1. Notice event (Attention)
          2. Check AI recommendation (Trust)
          3. Verify data? (Effort vs. Value)
          4. Decide action (Execution)
        """
        ai_pred = event.get("ai_prediction", {})
        ai_risk = ai_pred.get("risk_level", "low")
        ai_conf = ai_pred.get("confidence", 0.0)
        true_risk = event["risk_level"]
        
        # 1. Attention & Fatigue
        # Fatigue increases error probability
        effective_error_rate = self.profile["base_error_rate"] + self.fatigue_level
        self.fatigue_level += self.profile.get("fatigue_growth", 0.0)
        
        # 2. Verification Decision
        # Will the operator verify the AI's claim or just click "Accept"?
        # Influenced by profile propensity + UI scaffolding
        p_verify = self.profile["verification_propensity"] + self.ui["verification_boost"]
        
        # Lower verification if trust is high and AI is confident (complacency)
        if self.trust_in_ai > 0.8 and ai_conf > 0.9:
            p_verify *= 0.6
            
        is_verified = RNG.random() < p_verify
        
        # 3. Decision Logic
        if is_verified:
            # Operator looks at evidence (min_range, prob, stale data)
            # Simulates "finding the truth" but with some residual error chance
            if RNG.random() > effective_error_rate:
                # Correctly identifies the true risk based on physics rules
                perceived_risk = true_risk 
            else:
                 # Verification failed (human error) -> defaults to AI or random
                 perceived_risk = ai_risk
        else:
            # No verification -> Rely on AI prediction (Automation Bias)
            perceived_risk = ai_risk
            
        # 4. Action Execution
        # Map perceived risk to action
        from src.config import ACTION_RULES
        chosen_action = ACTION_RULES.get(perceived_risk, "log_only")
        
        # UI friction/interlock (e.g., Variant C checklist for critical)
        time_taken = self.profile["decision_time_mean"]
        if is_verified:
            time_taken += self.ui["time_cost_evidence"]
            
        if self.ui["checklist_required"] and perceived_risk in ["critical", "high"]:
            # Checklists reduce error but take time
            if RNG.random() < self.profile["checklist_adherence"]:
                # Checklist caught an error? (if any)
                pass 
            time_taken += self.ui["time_cost_checklist"]

        return {
            "event_id": event["event_id"],
            "true_risk": true_risk,
            "ai_risk": ai_risk,
            "operator_risk": perceived_risk,
            "action_taken": chosen_action,
            "action_correct": chosen_action == event["action_true"],
            "verification_performed": is_verified,
            "time_taken": time_taken
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
                    if res["true_risk"] == "critical" and res["action_taken"] != "execute_maneuver":
                        c = COST_WEIGHTS["missed_critical"]
                        shift_miss += 1
                    elif res["true_risk"] == "high" and res["action_taken"] not in ["execute_maneuver", "prepare_maneuver"]:
                        c = COST_WEIGHTS["missed_high"]
                        shift_miss += 1
                    elif res["action_taken"] == "execute_maneuver" and res["true_risk"] not in ["critical", "high"]:
                        c = COST_WEIGHTS["false_maneuver"]
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
