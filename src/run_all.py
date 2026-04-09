"""
Main execution pipeline for ORBIT-GUARD.

Steps:
  1. Fetch real conjunction data (SOCRATES)
  2. Train AI Risk Model (Random Forest)
  3. Run Operator Simulation (Monte Carlo)
  4. Export data for Web UI
"""

import os
import sys
import json
import time
import pandas as pd

# Allow running as `python src/run_all.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conjunction_data import load_conjunctions, export_for_web
from src.risk_model import train_risk_model, predict_risk
from src.operator_sim import run_simulation
from src.config import TABLES_DIR, N_SIMULATION_RUNS


def run_pipeline():
    start_time = time.time()
    
    # 1. Load Data
    print("Step 1: Loading SOCRATES conjunction data...")
    events = load_conjunctions()
    print(f"  Loaded {len(events)} events.")
    
    # 2. Train AI Model
    print("\nStep 2: Training AI Risk Model...")
    metrics = train_risk_model(events)
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print("  Feature Importance:")
    for k, v in sorted(metrics["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"    - {k}: {v:.3f}")
        
    # 3. Filter & Predict (Optimization)
    print("\nStep 3: Filtering & Generating AI predictions...")
    
    # Keep all critical/high risk, plus top medium/low by probability
    # Sorting by max_probability descending
    events.sort(key=lambda x: x["max_probability"], reverse=True)
    
    # Take top 2000 events for the UI/Simulation to keep things fast
    # (114k is too many for the browser/sim loop)
    events_subset = events[:2000]
    print(f"  Subset selected: {len(events_subset)} events (top risk)")
    
    events_with_ai = predict_risk(events_subset)
    
    # 4. Run Simulation
    # Sample 500 from the subset
    sim_events = events_with_ai[:500]
            
    print(f"\nStep 4: Running Monte Carlo Simulation ({N_SIMULATION_RUNS} runs on {len(sim_events)} sampled events)...")
    sim_start = time.time()
    results = run_simulation(sim_events, n_runs=N_SIMULATION_RUNS)
    print(f"  Simulation completed in {time.time() - sim_start:.1f}s")
    
    # 5. Export Results
    print("\nStep 5: Exporting data...")
    
    # Web UI export
    web_path = export_for_web(events_with_ai, "conjunctions.json")
    print(f"  UI Data -> {web_path}")
    
    # Simulation tables
    sim_df = pd.DataFrame(results)
    sim_csv = os.path.join(TABLES_DIR, "simulation_results.csv")
    sim_df.to_csv(sim_csv, index=False)
    print(f"  Sim Results -> {sim_csv}")
    
    # Summary
    print("\n--- Pipeline Summary ---")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Events processed: {len(events)}")
    print(f"Accuracy (Baseline A): {sim_df[sim_df['variant']=='A']['accuracy_mean'].mean():.2%}")
    print(f"Accuracy (Focus C):    {sim_df[sim_df['variant']=='C']['accuracy_mean'].mean():.2%}")


if __name__ == "__main__":
    run_pipeline()
