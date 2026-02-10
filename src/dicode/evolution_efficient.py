"""Asynchronous task evolution utilities for DiCode.

This module provides functions for:
1. Running evolution workers in background threads.
2. Task activation via compare-and-swap.
3. Validation of generated tasks.
"""

# --- Standard Library ---
import concurrent.futures
import traceback

# --- Third-Party ---
import jax
import numpy as np
from flax.training.train_state import TrainState
from omegaconf import DictConfig

# --- Local Modules ---
from dicode.dreaming.gen_manager import GenManager, TaskArchive
from dicode.logging_utils import log_wandb_table_append
from dicode.runtime_analysis import tracker
from dicode.selection import select_tasks_for_evolution


def attempt_to_activate_task(
    gen_manager: GenManager,
    new_task_id: str,
    new_task_score: float,
    config: DictConfig,
) -> bool:
    """Attempts to insert a new task into the active set via compare-and-swap.

    If the active set is not full, compares against the minimum entry threshold.
    If full, compares against the worst active task and evicts if better.

    Args:
        gen_manager: The GenManager instance.
        new_task_id: ID of the new task to activate.
        new_task_score: Priority score of the new task.
        config: Hydra configuration.

    Returns:
        True if the task was activated, False otherwise.
    """
    active_count = gen_manager.archive.active_task_count
    capacity = config.dicode_manager.active_task_capacity

    score_to_beat = -np.inf
    worst_task_id = None

    if active_count < capacity:
        # Active set not full: compare against threshold
        score_to_beat = config.dicode_manager.min_entry_score_threshold
    else:
        # Active set full: find and compare against worst task
        worst_task_id, worst_score = _find_worst_active_task(gen_manager.archive)
        score_to_beat = worst_score

    # Compare and swap
    if new_task_score >= score_to_beat:
        if worst_task_id is not None:
            gen_manager.archive.set_task_active_status(worst_task_id, False)
        gen_manager.archive.set_task_active_status(new_task_id, True)
        return True
    else:
        return False


def _find_worst_active_task(archive: TaskArchive) -> tuple[str | None, float]:
    """Finds the active task with the lowest priority score.

    Args:
        archive: The task archive.

    Returns:
        Tuple of (task_id, score) for the worst task, or (None, inf) if none found.
    """
    worst_id = None
    worst_score = np.inf

    with archive._lock:
        graph_nodes = list(archive.graph.nodes(data=True))

    for node_id, data in graph_nodes:
        if data.get("is_active"):
            score = data.get("priority_score", 0.0)
            if score < worst_score:
                worst_score = score
                worst_id = node_id

    return (worst_id, worst_score) if worst_id else (None, np.inf)


def dispatch_evolution_worker(
    executor: concurrent.futures.ThreadPoolExecutor,
    evolve_future: concurrent.futures.Future | None,
    gen_manager: GenManager,
    config: DictConfig,
    evaluation_metrics: dict,
) -> concurrent.futures.Future | None:
    """Dispatches a new evolution worker if one is not already running.

    This function:
    1. Selects eligible tasks for evolution.
    2. Submits an I/O-bound job to the thread pool.
    3. Returns the Future tracking the job.

    Args:
        executor: Thread pool executor.
        evolve_future: Existing future (should be None to dispatch).
        gen_manager: The GenManager instance.
        config: Hydra configuration.
        evaluation_metrics: Current evaluation metrics.

    Returns:
        The Future for the evolution job, or None if no tasks to evolve.
    """
    if evolve_future is not None:
        return evolve_future

    print("  Selecting tasks and dispatching evolution job...")

    # Select tasks for evolution
    num_to_evolve = (
        config.dicode_manager.num_generation_tasks + config.dicode_manager.additional_num_parents
    )
    tasks_to_evolve = select_tasks_for_evolution(
        config, gen_manager.archive, gen_manager.session_idx - 1, num_to_evolve
    )

    if not tasks_to_evolve:
        print("  No eligible tasks to evolve.")
        return None

    # Submit evolution job
    evolve_future = executor.submit(
        evolve_and_validate_tasks,
        gen_manager,
        {"mastered": tasks_to_evolve, "unlearnable": []},
        evaluation_metrics,
        gen_manager.session_idx,
    )
    print(f"  Dispatched evolution job for {len(tasks_to_evolve)} tasks.")

    return evolve_future


def evolve_and_validate_tasks(
    gen_manager: GenManager,
    tasks: dict,
    metrics: dict,
    session_idx: int,
) -> list[dict]:
    """Runs evolution and validation in the background worker.

    This function:
    1. Calls gen_manager.evolve_tasks (LLM generation).
    2. Runs check_compilation on each generated task in parallel.
    3. Returns results with compilation status.

    Args:
        gen_manager: The GenManager instance.
        tasks: Dictionary of tasks to evolve.
        metrics: Current evaluation metrics.
        session_idx: Current session index.

    Returns:
        List of task generation results with compilation status.
    """
    tracker.start_timer("Evolution (LLM)")
    generation_results = gen_manager.evolve_tasks(tasks, metrics)
    tracker.stop_timer("Evolution (LLM)", session_idx)

    if not generation_results:
        return []

    print(f"    WORKER: Validating {len(generation_results)} generated tasks...")

    # Validate in parallel
    tracker.start_timer("Compilation (LLM)")
    with concurrent.futures.ThreadPoolExecutor() as validation_executor:
        future_to_task = {
            validation_executor.submit(
                gen_manager.env_generator.check_compilation, res["code_string"]
            ): res
            for res in generation_results
            if res.get("code_string")
        }

        for future in concurrent.futures.as_completed(future_to_task):
            res = future_to_task[future]
            try:
                is_correct, error_msg = future.result()
                res["compiled"] = is_correct
                res["error"] = None if is_correct else error_msg
            except Exception as e:
                res["compiled"] = False
                res["error"] = f"Validation exception: {e}"

    print("    WORKER: Validation complete.")
    tracker.stop_timer("Compilation (LLM)", session_idx)
    return generation_results


def run_session_validation(
    config: DictConfig,
    rng: jax.Array,
    rl_train_state: TrainState,
    gen_manager: GenManager,
    evolve_future: concurrent.futures.Future | None,
    current_session_idx: int,
) -> tuple:
    """Checks for results from the evolution worker and runs validation.

    Args:
        config: Hydra configuration.
        rng: JAX PRNG key.
        rl_train_state: Current agent state.
        gen_manager: The GenManager instance.
        evolve_future: Future from the evolution worker.
        current_session_idx: Current session index.

    Returns:
        Tuple of (rng, evolve_future, activated_count, rejected_count,
                  generation_table, new_task_ids).
    """
    activated_count = 0
    rejected_count = 0
    generation_table = None
    new_task_ids = []

    if evolve_future is None or not evolve_future.done():
        return rng, evolve_future, 0, 0, None, []

    print("  Worker finished. Retrieving results...")
    try:
        worker_results = evolve_future.result()
        evolve_future = None

        if worker_results:
            rng, validation_rng = jax.random.split(rng)
            validation_metrics = run_validation_on_main_thread(
                config, validation_rng, rl_train_state, gen_manager, worker_results
            )

            activated_count = validation_metrics.get("activated_count", 0)
            rejected_count = validation_metrics.get("rejected_count", 0)
            new_task_ids = validation_metrics.get("new_task_ids", [])

            if config.use_wandb and worker_results:
                generation_table = _log_generation_results(
                    worker_results, current_session_idx
                )
        else:
            print("  Worker returned no new tasks.")

    except Exception as e:
        print(f"  ERROR: Evolution worker failed: {e}")
        traceback.print_exc()
        evolve_future = None

    return rng, evolve_future, activated_count, rejected_count, generation_table, new_task_ids


def run_validation_on_main_thread(
    config: DictConfig,
    rng: jax.Array,
    train_state: TrainState,
    gen_manager: GenManager,
    worker_results: list[dict],
) -> dict:
    """Runs validation on the main thread for compiled tasks.

    Args:
        config: Hydra configuration.
        rng: JAX PRNG key.
        train_state: Current agent state.
        gen_manager: The GenManager instance.
        worker_results: Results from the evolution worker.

    Returns:
        Dictionary with activated_count, rejected_count, and new_task_ids.
    """
    activated_count = 0
    rejected_count = 0
    new_task_ids = []

    if not worker_results:
        return {"activated_count": 0, "rejected_count": 0, "new_task_ids": []}

    print(f"  Validating {len(worker_results)} new task designs...")

    for res in worker_results:
        task_id = res.get("generated_task_id")
        code_string = res.get("code_string")
        reasoning = res.get("reasoning")

        if not task_id or not code_string:
            rejected_count += 1
            if task_id:
                gen_manager.archive.update_node_status(task_id, "compile_fail_no_code")
            continue

        if reasoning:
            gen_manager.archive.update_node_reasoning(task_id, reasoning)

        # Check compilation
        is_correct, error_msg = gen_manager.env_generator.check_compilation(code_string)

        if is_correct:
            gen_manager.archive.update_node_status(task_id, "compiled")
            gen_manager.archive.set_task_active_status(task_id, False)
            gen_manager.archive.update_node_priority_score(task_id, 0.0)
            new_task_ids.append(task_id)
            activated_count += 1
            res["compiled"] = True
        else:
            rejected_count += 1
            res["compiled"] = False
            res["error"] = error_msg

    return {
        "activated_count": activated_count,
        "rejected_count": rejected_count,
        "new_task_ids": new_task_ids,
    }


def _log_generation_results(
    worker_results: list[dict], current_session_idx: int
) -> "wandb.Table":
    """Logs generation results to W&B table."""
    columns = [
        "session_idx",
        "evolution_type",
        "parent_task_id",
        "parent_task",
        "generated_task_id",
        "compilation_status",
        "LLM_reasoning",
        "generated_docstring",
        "final_code",
    ]

    rows = []
    for result in worker_results:
        rows.append([
            current_session_idx,
            result.get("evolution_type"),
            result.get("parent_task_id"),
            result.get("parent_task"),
            result.get("generated_task_id"),
            "✅ Success" if result.get("compiled") else "❌ Fail",
            result.get("reasoning"),
            result.get("docstring"),
            result.get("code"),
        ])

    return log_wandb_table_append(
        table_key="curriculum/generation_table",
        columns=columns,
        new_data_rows=rows,
    )
