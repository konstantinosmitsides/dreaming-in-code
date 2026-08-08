"""Evaluate trained checkpoints on the held-out Craftax test set.

This is the harness that produced every number in the paper. All methods -- DiCode, PPO-GTrXL,
PLR, SFL and DR -- are scored by this one script, from the same fixed `jax.random.PRNGKey(0)`,
on `CraftaxAugObsTrain` over `--num-envs` procedurally generated worlds. The paper used
1024 worlds x 8192 steps.

It expects a directory tree of checkpoints laid out as:

    <results-dir>/<METHOD>/<SEED>/rl_checkpoints/<UPDATE_STEP>/

Each (method, seed, step) is evaluated once and cached as JSON in `--cache-dir`, so the scan is
resumable; `--aggregate-only` collects an existing cache into a dataframe without re-evaluating.

    python experiments/evaluation/run_paper_evaluation.py \\
        --results-dir path/to/results --cache-dir path/to/cache --output results.pkl
"""

import argparse
import os
import json
import glob
import re
import jax
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from dicode.paper_evaluation.craftax_checkpoint_tr_evaluation import evaluate_checkpoint

# ==========================================
# 1. Configuration
# ==========================================

# Set from the CLI in __main__; see parse_args().
ROOT_DIR = None
CACHE_DIR = None
FINAL_OUTPUT_FILE = None

def get_env_steps(method_name, update_step):
    """
    CONVERSION LOGIC: update_step -> environment_step
    
    Adjust the multipliers below based on your specific algorithms.
    """

    # Convert string input to integer
    step = int(update_step)
    
    if "sfl" in method_name.lower():
        # Example: PPO usually has (num_envs * num_steps) per update
        # If num_envs=128, num_steps=128 -> 16384 env steps per update
        return step * 271697

    else:
        return step * 1024 * 128

# ==========================================
# 2. The Scanner & Cacher
# ==========================================

def get_cache_path(method, seed, step):
    """Generates a unique filename for the cached result."""
    # sanitizing names just in case
    safe_method = "".join([c for c in method if c.isalnum() or c in ('_', '-')])
    filename = f"{safe_method}_seed{seed}_step{step}.json"
    return os.path.join(CACHE_DIR, filename)

def load_config_from_folder(checkpoint_path):
    """
    Attempts to load .hydra/config.yaml or similar from the checkpoint folder.
    Adapt this if your configs are stored differently.
    """

    dir_path = os.path.abspath(os.path.join(checkpoint_path, "..", ".."))

    config_path = os.path.join(dir_path, ".hydra", "config.yaml")
    json_config_path = os.path.join(dir_path, "config.json")
    if not os.path.exists(config_path):
        # Fallback: look one level up (in seed dir) or just default
        config_path = os.path.join(dir_path, "..", ".hydra", "config.yaml")
    
    if os.path.exists(config_path):
        return OmegaConf.load(config_path)
    elif os.path.exists(json_config_path):
        with open(json_config_path, "r") as f:
            flat_config = json.load(f)

        flat_config["condition_on_task"] = False
        flat_config["anneal_lr"] = False
        return OmegaConf.create({"training": flat_config})
    else:
        raise FileNotFoundError(f"Could not find config.yaml in {folder_path} or parent.")

def scan_and_evaluate(num_envs=1024, num_steps=8192):
    """
    Walks through ROOT_DIR, finds valid checkpoints, and runs evaluation
    if not already cached.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    # Regex to extract method, seed, update_step from path
    # Assuming structure: paper_results/method_name/seed/update_step
    # Note: 'seed' and 'update_step' folders might be named '1', '100', etc.
    
    jobs = []
    
    print(f"Scanning {ROOT_DIR} for checkpoints...")
    
    # 1. SCAN PHASE
    for method_name in os.listdir(ROOT_DIR):
        method_path = os.path.join(ROOT_DIR, method_name)
        if not os.path.isdir(method_path): continue
            
        for seed_name in os.listdir(method_path):
            seed_path = os.path.join(method_path, seed_name)
            if not os.path.isdir(seed_path): continue

            checkpoints_dir = os.path.join(seed_path, "rl_checkpoints")

            if not os.path.isdir(checkpoints_dir): continue


                
            for step_name in os.listdir(checkpoints_dir):
                step_path = os.path.join(checkpoints_dir, step_name)
                # Check if this is actually a checkpoint folder (contains weights)
                # Adjust 'params.pkl' or 'checkpoint' to whatever file indicates a valid save
                if not os.path.isdir(step_path): continue
                
                # Verify it looks like a checkpoint (optional but safer)
                # if not os.path.exists(os.path.join(step_path, 'checkpoint')): continue

                abs_step_path = os.path.abspath(step_path)

                jobs.append({
                    "method": method_name,
                    "seed": seed_name,
                    "update_step": step_name,
                    "full_path": abs_step_path
                })

    print(f"Found {len(jobs)} potential checkpoints.")
    
    # 2. EXECUTION PHASE
    
    for i, job in enumerate(jobs):
        method = job["method"]
        seed = job["seed"]
        u_step = job["update_step"]
        path = job["full_path"]
        
        cache_file = get_cache_path(method, seed, u_step)
        
        # --- THE "CLEVER" PART ---
        if os.path.exists(cache_file):
            print(f"[{i+1}/{len(jobs)}] Skipping {method}/{seed}/{u_step} (Already cached)")
            continue
            
        print(f"[{i+1}/{len(jobs)}] Evaluating {method}/{seed}/{u_step} ...")
        
        try:
            # Load Config
            # (Assuming you need the config object for the evaluate function)
            # If you are passing the config manually in main, adapt this.
            config = load_config_from_folder(path) 
            
            init_rng = jax.random.PRNGKey(0)            
            # --- CALL YOUR EVALUATION FUNCTION ---
            # Returns a dictionary of metrics
            metrics = evaluate_checkpoint(
                method=method,
                checkpoint_path=path,
                num_envs=num_envs,
                num_steps=num_steps,
                rng=init_rng,
                config=config
            )
            
            # --- POST PROCESS ---
            # 1. Convert JAX arrays to python lists/floats for JSON serialization
            # This is crucial because standard JSON cannot dump JAX/Numpy arrays
            clean_metrics = {}
            for k, v in metrics.items():
                if hasattr(v, 'tolist'):
                    clean_metrics[k] = v.tolist()
                elif isinstance(v, (np.generic, jax.Array)):
                    clean_metrics[k] = v.item()
                else:
                    clean_metrics[k] = v
            
            # 2. Add Metadata
            env_steps = get_env_steps(method, u_step)
            
            result_payload = {
                "method": method,
                "seed": seed,
                "update_step": int(u_step),
                "env_steps": env_steps,
                **clean_metrics
            }
            
            # 3. Save to Cache
            with open(cache_file, 'w') as f:
                json.dump(result_payload, f)
                
            print(f"   -> Saved to {cache_file}")
            
        except Exception as e:
            print(f"   [ERROR] Failed to evaluate {path}: {e}")
            import traceback
            traceback.print_exc()

# ==========================================
# 3. The Aggregator
# ==========================================

def aggregate_and_save():
    print("\n--- Aggregating Results ---")
    data_list = []
    
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*.json"))
    print(f"Found {len(cache_files)} cached files.")
    
    for cf in cache_files:
        try:
            with open(cf, 'r') as f:
                data = json.load(f)
                data_list.append(data)
        except Exception as e:
            print(f"Error reading {cf}: {e}")
            
    if not data_list:
        print("No data to aggregate.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data_list)
    
    # Sort for tidiness
    df.sort_values(by=["method", "env_steps", "seed"], inplace=True)
    
    # Save
    out_dir = os.path.dirname(FINAL_OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_pickle(FINAL_OUTPUT_FILE)
    print(f"Successfully saved aggregated results to {FINAL_OUTPUT_FILE}")
    print("Columns available:", df.columns.tolist())
    
    # Preview
    print("\nTop 5 rows:")
    print(df[["method", "env_steps", "mean_return"]].head())

# ==========================================
# 4. Main Entry Point
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                        help="Root of the checkpoint tree: <results-dir>/<METHOD>/<SEED>/rl_checkpoints/<STEP>/")
    parser.add_argument("--cache-dir", required=True,
                        help="Where per-checkpoint JSON results are cached. Re-runs skip what is already here.")
    parser.add_argument("--output", required=True,
                        help="Path for the aggregated dataframe (pickle).")
    parser.add_argument("--num-envs", type=int, default=1024,
                        help="Held-out worlds to evaluate on. The paper used 1024.")
    parser.add_argument("--num-steps", type=int, default=8192,
                        help="Steps per episode; must exceed the environment's max_timesteps. The paper used 8192.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip evaluation and just collect an existing cache into --output.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ROOT_DIR = args.results_dir
    CACHE_DIR = args.cache_dir
    FINAL_OUTPUT_FILE = args.output

    if not args.aggregate_only:
        scan_and_evaluate(num_envs=args.num_envs, num_steps=args.num_steps)

    aggregate_and_save()