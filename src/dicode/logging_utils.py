"""Logging utilities for DiCode.

This module provides functions for:
1. Logging session summaries to W&B.
2. Appending data to W&B tables with persistence.
"""

# --- Standard Library ---
import json
import os

# --- Third-Party ---
import numpy as np
import wandb
from omegaconf import DictConfig

# --- Local Modules ---
from dicode.dreaming.gen_manager import GenManager
from dicode.utils.logz.graph_visualization import create_graph_visualization_html


def log_session_summary(
    config: DictConfig,
    current_session_idx: int,
    global_env_steps: int,
    global_update_step: int,
    gen_manager: GenManager,
    sampled_task_ids: list[str],
    num_updates_in_session: int,
    training_metrics: dict,
    categorized_tasks: dict,
    generation_table: "wandb.Table | None",
    rl_ckpt_path: str,
    worker_wait_time: float = 0.0,
    worker_total_time: float = 0.0,
    cumulative_compiled: int = 0,
    cumulative_activated: int = 0,
) -> None:
    """Logs all metrics and artifacts for a completed session to W&B.

    Args:
        config: Hydra configuration.
        current_session_idx: Current session index.
        global_env_steps: Total environment steps.
        global_update_step: Total training updates.
        gen_manager: The GenManager instance.
        sampled_task_ids: Task IDs sampled for training.
        num_updates_in_session: Updates completed in this session.
        training_metrics: Per-task training metrics.
        categorized_tasks: Tasks categorized by status.
        generation_table: W&B table for generation results.
        rl_ckpt_path: Path to agent checkpoints.
        worker_wait_time: Time spent waiting for evolution worker.
        worker_total_time: Total evolution worker execution time.
        cumulative_compiled: Total tasks compiled so far.
        cumulative_activated: Total tasks activated so far.
    """
    if not config.use_wandb:
        return

    print("--- [Main Thread] Logging session summary... ---")

    # Core metrics
    log_data = {
        "session": current_session_idx,
        "global_env_steps": global_env_steps,
        "global_update_step": global_update_step,
        # System metrics
        "system/worker_wait_time_seconds": worker_wait_time,
        "system/worker_execution_time_seconds": worker_total_time,
        "system/efficiency_ratio": (
            worker_wait_time / worker_total_time if worker_total_time > 0 else 0
        ),
        # Curriculum metrics
        "curriculum/num_tasks_activated_cumulative": cumulative_activated,
        "curriculum/num_tasks_compiled_cumulative": cumulative_compiled,
        "curriculum/activation_success_ratio": (
            cumulative_activated / cumulative_compiled if cumulative_compiled > 0 else 0.0
        ),
        "curriculum/num_tasks_sampled": len(sampled_task_ids),
        # Training metrics
        "training/updates_done_in_session": float(num_updates_in_session),
    }

    # Add training averages
    if training_metrics:
        _add_training_averages(log_data, training_metrics, categorized_tasks)

    # Add archive statistics
    _add_archive_statistics(log_data, gen_manager)

    # Add graph visualization
    _add_graph_visualization(log_data)

    wandb.log(log_data)
    print("  Session summary logged to W&B.")


def _add_training_averages(
    log_data: dict, training_metrics: dict, categorized_tasks: dict
) -> None:
    """Adds training average metrics to log data."""
    all_srs = [m["sr"] for m in training_metrics.values() if m.get("sr", -1.0) >= 0]
    all_lps = [
        m["lp"] for m in training_metrics.values() if not np.isnan(m.get("lp", np.nan))
    ]

    log_data["training/avg_sr_trained"] = np.mean(all_srs) if all_srs else 0.0
    log_data["training/avg_lp_trained"] = np.mean(all_lps) if all_lps else 0.0

    # Status counts
    for status in ["A", "B", "C", "D"]:
        log_data[f"curriculum/num_newly_{status}"] = len(
            categorized_tasks.get(status, [])
        )


def _add_archive_statistics(log_data: dict, gen_manager: GenManager) -> None:
    """Adds archive statistics to log data."""
    all_nodes = gen_manager.archive.graph.nodes(data=True)
    total_nodes = len(all_nodes)

    status_counts = {s: 0 for s in ["A", "B", "C", "D"]}
    for _, data in all_nodes:
        status = data.get("status")
        if status in status_counts:
            status_counts[status] += 1

    total_trainable = sum(status_counts.values())

    log_data["curriculum/archive_total_tasks"] = total_nodes
    log_data["curriculum/archive_total_trainable_tasks"] = total_trainable

    for status, count in status_counts.items():
        pct = (count / total_trainable * 100.0) if total_trainable > 0 else 0.0
        log_data[f"curriculum/archive_{status}_pct"] = pct


def _add_graph_visualization(log_data: dict) -> None:
    """Adds interactive graph visualization to log data."""
    graphml_path = "task_graph.graphml"
    try:
        graph_html = create_graph_visualization_html(graphml_path)
        log_data["curriculum/interactive_graph"] = wandb.Html(graph_html)
    except Exception as e:
        print(f"  Warning: Could not generate graph visualization. {e}")


def log_wandb_table_append(
    table_key: str, columns: list[str], new_data_rows: list[list]
) -> "wandb.Table | None":
    """Appends data to a W&B table with local persistence.

    This ensures table data persists across resumed runs by storing
    the full history in a local JSON file.

    Args:
        table_key: The key to log to W&B (e.g., "curriculum/task_performance").
        columns: Column names for the table.
        new_data_rows: New rows to append.

    Returns:
        The combined wandb.Table, or None if no data.
    """
    if not new_data_rows:
        print(f"  No new data for table '{table_key}'.")
        return None

    # Local file for persistence
    safe_filename = table_key.replace("/", "_") + ".json"
    json_log_file = os.path.join(wandb.run.dir, safe_filename)

    # Load existing data
    try:
        with open(json_log_file) as f:
            all_data = json.load(f)
        print(f"  Loaded {len(all_data)} existing rows for '{table_key}'.")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"  Creating new local log for '{table_key}'.")
        all_data = []

    # Append new rows
    all_data.extend(new_data_rows)

    # Save combined data
    try:
        with open(json_log_file, "w") as f:
            json.dump(all_data, f)
    except Exception as e:
        print(f"  Warning: Could not save local table log. {e}")

    return wandb.Table(columns=columns, data=all_data)
