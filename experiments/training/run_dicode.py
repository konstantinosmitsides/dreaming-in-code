"""DiCode: Main training script for online curriculum learning.

This script orchestrates the DiCode training loop, which interleaves:
1. Task evolution (LLM-based generation of new tasks)
2. Training (PPO on a sampled batch of tasks)
3. Task activation (compare-and-swap for the active set)
"""

# --- Standard Library ---
import concurrent.futures
import gc
import os
import random
import time

# --- Third-Party ---
import hydra
import jax
import wandb
from omegaconf import DictConfig

# --- Local Modules ---
from dicode.evaluation import run_session_evaluation
from dicode.evolution_efficient import (
    attempt_to_activate_task,
    dispatch_evolution_worker,
)
from dicode.logging_utils import log_session_summary
from dicode.runtime_analysis import tracker
from dicode.selection import sample_tasks_for_training
from dicode.setup import run_initial_seed_training, setup_experiment
from dicode.training import run_session_training


# --- Constants ---
MAX_JAX_CACHE_CLEAR_RETRIES = 10


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="config")
def main(config: DictConfig):
    """Main entry point for the DiCode training loop."""

    # =========================================================================
    # Phase 1: Experiment Setup
    # =========================================================================
    (
        rng,
        gen_manager,
        rl_ckpt_manager,
        rl_train_state,
        global_update_step,
        global_env_steps,
        latest_step,
        cumulative_compiled,
        cumulative_activated,
    ) = setup_experiment(config)

    rl_ckpt_path = os.path.join(os.getcwd(), config.checkpoint_dir)
    last_known_original_return = 0.0

    # =========================================================================
    # Phase 1.5: Initial Seed Training (only if starting from scratch)
    # =========================================================================
    needs_seed_training = gen_manager.session_idx <= 1 and latest_step is None

    if needs_seed_training:
        (
            rng,
            rl_train_state,
            global_update_step,
            global_env_steps,
            evaluation_metrics,
        ) = run_initial_seed_training(
            config,
            gen_manager,
            rng,
            rl_train_state,
            global_update_step,
            global_env_steps,
        )
        global_env_steps = int(global_env_steps)
        last_known_original_return = evaluation_metrics.get("mean_return", 0.0)

    elif latest_step is not None:
        print("Agent state loaded from checkpoint, skipping initial seed training.")
        # Run one-off evaluation to prime metrics for the first evolution step
        rng, evaluation_metrics = run_session_evaluation(
            config,
            rng,
            rl_train_state,
            gen_manager,
            gen_manager.session_idx,
            global_env_steps,
        )
        last_known_original_return = evaluation_metrics.get("mean_return", 0.0)
        print(f"Original return: {last_known_original_return}")

    # =========================================================================
    # Phase 2: Initialize Background Evolution Worker
    # =========================================================================
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    evolve_future = None
    worker_start_time = 0.0
    print("Initialized ThreadPoolExecutor for background evolution.")

    # Evolution interval: how many sessions to wait before syncing with worker.
    # If k=2: Train -> Hold -> Sync & Train with new tasks
    evolution_interval = config.dicode_manager.get("evolution_interval", 2)
    sessions_since_evolution = evolution_interval  # Start with sync on first loop

    # =========================================================================
    # Phase 3: Main Curriculum Loop
    # =========================================================================
    while global_env_steps < config.training.total_timesteps:
        current_session_idx = gen_manager.session_idx
        print(f"\n{'=' * 60}")
        print(f"--- Starting Session {current_session_idx} ---")
        print(f"{'=' * 60}")

        # --- Step 1: Check if we should sync with evolution worker ---
        new_task_ids = []
        compiled_count = 0
        generation_table = None
        current_worker_wait_time = 0.0
        current_worker_total_time = 0.0

        should_sync = sessions_since_evolution >= evolution_interval

        if should_sync:
            print(f"  [Sync] Iteration {evolution_interval} reached. Waiting for worker...")

            if evolve_future is not None:
                wait_start = time.time()
                worker_results = evolve_future.result()  # Blocks until done
                wait_end = time.time()

                current_worker_wait_time = wait_end - wait_start
                current_worker_total_time = wait_end - worker_start_time
                evolve_future = None

                print(f"  [Timing] Waited: {current_worker_wait_time:.2f}s | "
                      f"Total: {current_worker_total_time:.2f}s")

                new_task_ids, compiled_count = _process_worker_results(
                    worker_results, gen_manager, config
                )

            sessions_since_evolution = 1
        else:
            print(f"  [Hold] Iteration {sessions_since_evolution}/{evolution_interval}. "
                  "Skipping new tasks.")
            sessions_since_evolution += 1

        # --- Step 2: Dispatch new evolution worker if needed ---
        if evolve_future is None:
            worker_start_time = time.time()
            evolve_future = dispatch_evolution_worker(
                executor, evolve_future, gen_manager, config, evaluation_metrics
            )

        # --- Step 3: Sample tasks for training ---
        print("Sampling tasks for training...")
        target_batch_size = config.dicode_manager.training_sample_size_n
        num_new_to_use = len(new_task_ids)
        num_to_sample_from_archive = max(0, (target_batch_size - 1) - num_new_to_use)

        print(f"  New tasks: {num_new_to_use}. "
              f"Sampling {num_to_sample_from_archive} from archive.")

        sampled_from_archive = sample_tasks_for_training(
            gen_manager, config, num_to_sample_from_archive
        )
        sampled_task_ids = new_task_ids + sampled_from_archive

        if not sampled_task_ids:
            print("  No tasks sampled. Skipping to next session.")
            gen_manager.session_idx += 1
            continue

        # --- Step 4: Run Training ---
        tracker.start_timer("Training")
        (
            rng,
            rl_train_state,
            global_update_step,
            global_env_steps,
            training_metrics,
            num_updates_in_session,
            categorized_tasks,
            evaluation_metrics,
        ) = run_session_training(
            config,
            rng,
            rl_train_state,
            gen_manager,
            global_update_step,
            global_env_steps,
            current_session_idx,
            sampled_task_ids,
            original_return_prev_session=last_known_original_return,
        )
        global_env_steps = int(global_env_steps)

        if "mean_return" in evaluation_metrics:
            last_known_original_return = evaluation_metrics["mean_return"]
            print(f"  Updated Original Task Return: {last_known_original_return:.2f}")

        # Log evaluation metrics
        if config.use_wandb and evaluation_metrics:
            eval_log_data = {
                "session": current_session_idx,
                "global_env_steps": global_env_steps,
            }
            for key, value in evaluation_metrics.items():
                eval_log_data[f"evaluation/{key}"] = value
            wandb.log(eval_log_data)

        # --- Step 5: Post-Training Activation (Compare-and-Swap) ---
        real_activated_count = 0
        if new_task_ids:
            print(f"  Attempting to activate {len(new_task_ids)} new tasks...")
            for new_task_id in new_task_ids:
                with gen_manager.archive._lock:
                    if gen_manager.archive.graph.has_node(new_task_id):
                        new_score = gen_manager.archive.graph.nodes[new_task_id].get(
                            "priority_score", 0.0
                        )
                    else:
                        print(f"    Warning: Task {new_task_id} not found. Skipping.")
                        continue

                if attempt_to_activate_task(gen_manager, new_task_id, new_score, config):
                    real_activated_count += 1

        cumulative_compiled += compiled_count
        cumulative_activated += real_activated_count
        tracker.stop_timer("Training", current_session_idx)

        # Handle training failure
        if num_updates_in_session == 0 and not training_metrics and sampled_task_ids:
            print("  Error: Training session failed. Skipping to next session.")
            gen_manager.session_idx += 1
            continue

        # --- Step 6: Cleanup & Checkpointing ---
        print("Forcing garbage collection...")
        gc.collect()

        for i in range(MAX_JAX_CACHE_CLEAR_RETRIES):
            try:
                jax.clear_caches()
                break
            except RuntimeError as e:
                if "Set changed size" in str(e) and i < MAX_JAX_CACHE_CLEAR_RETRIES - 1:
                    time.sleep(0.1)
                    continue
                raise

        print("Checkpointing agent state and saving task graph...")
        rl_ckpt_manager.save(global_update_step, rl_train_state)
        gen_manager.archive.save_graph()

        # --- Step 7: Logging ---
        log_session_summary(
            config,
            current_session_idx,
            global_env_steps,
            global_update_step,
            gen_manager,
            sampled_task_ids,
            num_updates_in_session,
            training_metrics,
            categorized_tasks,
            generation_table,
            rl_ckpt_path,
            current_worker_wait_time,
            current_worker_total_time,
            cumulative_compiled=cumulative_compiled,
            cumulative_activated=cumulative_activated,
        )

        tracker.save_data()
        tracker.plot_results()

        gen_manager.session_idx += 1

    # =========================================================================
    # Phase 4: Final Cleanup
    # =========================================================================
    if config.use_wandb:
        print("\n--- Run complete. Closing W&B run. ---")
        wandb.finish()


def _process_worker_results(
    worker_results: list[dict] | None,
    gen_manager,
    config: DictConfig,
) -> tuple[list[str], int]:
    """Processes results from the evolution worker.

    Args:
        worker_results: List of task generation results from the worker.
        gen_manager: The GenManager instance managing the task archive.
        config: Hydra configuration.

    Returns:
        A tuple of (new_task_ids, compiled_count).
    """
    if not worker_results:
        return [], 0

    compiled_tasks = [res for res in worker_results if res.get("compiled")]
    failed_tasks = [res for res in worker_results if not res.get("compiled")]

    compiled_count = len(compiled_tasks)
    compile_fail_count = len(failed_tasks)

    print(f"  Worker returned {len(worker_results)} tasks: "
          f"{compiled_count} compiled, {compile_fail_count} failed.")

    # Selection: limit to configured number of tasks
    limit = config.dicode_manager.num_generation_tasks
    if compiled_count > limit:
        selected_new_tasks = random.sample(compiled_tasks, limit)
    else:
        selected_new_tasks = compiled_tasks

    # Register selected tasks in archive
    new_task_ids = []
    for res in selected_new_tasks:
        task_id = res.get("generated_task_id")
        code = res.get("code_string")
        reasoning = res.get("reasoning")

        if task_id and code:
            gen_manager.archive.update_node_status(task_id, "compiled")
            gen_manager.archive.set_task_active_status(task_id, False)
            gen_manager.archive.update_node_priority_score(task_id, 0.0)
            if reasoning:
                gen_manager.archive.update_node_reasoning(task_id, reasoning)
            gen_manager.archive.update_node_code(task_id, code)
            new_task_ids.append(task_id)

    return new_task_ids, compiled_count


if __name__ == "__main__":
    main()
