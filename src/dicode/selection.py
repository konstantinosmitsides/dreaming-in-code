"""Task selection utilities for DiCode.

This module provides functions for:
1. Sampling tasks for training from the active set (prioritized replay).
2. Selecting frontier tasks for evolution (parent selection).
"""

# --- Standard Library ---
import random

# --- Third-Party ---
import numpy as np
from omegaconf import DictConfig

# --- Local Modules ---
from dicode.dreaming.gen_manager import GenManager, TaskArchive


def sample_tasks_for_training(
    gen_manager: GenManager, config: DictConfig, n: int
) -> list[str]:
    """Samples n tasks from the active set for training using prioritized replay.

    Implements PLR-style sampling:
    1. Gets all tasks where `is_active=True`.
    2. Calculates score weight (w_s) based on rank or top-k prioritization.
    3. Calculates staleness weight (w_c) based on time since last trained.
    4. Combines: w = (1 - staleness_coeff) * w_s + staleness_coeff * w_c
    5. Samples `n` tasks using combined weights.

    Args:
        gen_manager: The GenManager instance.
        config: Hydra configuration.
        n: Number of tasks to sample.

    Returns:
        List of sampled task IDs.
    """
    print("  [Sampling] Starting prioritized sampling from active set...")

    # Get active tasks
    active_pool = []
    with gen_manager.archive._lock:
        for node_id, data in gen_manager.archive.graph.nodes(data=True):
            if data.get("is_active"):
                active_pool.append((node_id, data))

    if not active_pool:
        print("  [Sampling] Warning: No active tasks found.")
        return []

    current_session_idx = gen_manager.session_idx
    staleness_coeff = config.dicode_manager.staleness_coeff
    method = config.dicode_manager.prioritization_method
    temperature = config.dicode_manager.prioritization_temperature
    topk_k = config.dicode_manager.topk_k

    num_active = len(active_pool)
    print(f"  [Sampling] Method: {method} | Active tasks: {num_active}")

    # Extract data
    node_ids = [d[0] for d in active_pool]
    scores = np.array([d[1].get("priority_score", 0.0) for d in active_pool])

    # Calculate score weights
    w_s = _calculate_score_weights(scores, method, temperature, topk_k)

    # Calculate staleness weights
    timestamps = np.array([d[1].get("session_last_trained", 0) for d in active_pool])
    staleness = np.maximum(0.0, current_session_idx - timestamps)
    w_c = staleness / staleness.sum() if staleness.sum() > 0 else np.ones(num_active) / num_active

    # Combine weights
    final_weights = (1.0 - staleness_coeff) * w_s + staleness_coeff * w_c

    if final_weights.sum() <= 0:
        print("  [Sampling] Warning: Final weights are zero. Using uniform.")
        probabilities = np.ones(num_active) / num_active
    else:
        probabilities = final_weights / final_weights.sum()

    # Sample
    num_to_sample = min(n, num_active)
    sampled_task_ids = np.random.choice(
        node_ids, size=num_to_sample, replace=False, p=probabilities
    ).tolist()

    print(f"  [Sampling] Sampled {len(sampled_task_ids)} tasks.")
    return sampled_task_ids


def _calculate_score_weights(
    scores: np.ndarray,
    method: str,
    temperature: float,
    topk_k: int,
) -> np.ndarray:
    """Calculates score-based sampling weights.

    Args:
        scores: Array of priority scores for each task.
        method: Either "rank" or "topk".
        temperature: Temperature for rank-based softmax.
        topk_k: Number of top tasks for top-k method.

    Returns:
        Normalized weight array.
    """
    num_tasks = len(scores)

    if method == "rank":
        # Rank-based prioritization (from PLR)
        order = (-scores).argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(num_tasks) + 1
        w_s = (1.0 / ranks) ** (1.0 / temperature)

    elif method == "topk":
        k = min(num_tasks, topk_k)
        topk_indices = np.argpartition(-scores, k - 1)[:k]
        topk_mask = np.zeros(num_tasks, dtype=bool)
        topk_mask[topk_indices] = True

        # Softmax only on top-k scores
        scores_masked = np.where(topk_mask, scores, -np.inf)
        stable_scores = scores_masked - np.max(scores_masked)
        exp_scores = np.exp(stable_scores)
        exp_scores = np.where(topk_mask, exp_scores, 0.0)
        w_s = exp_scores

    else:
        raise ValueError(f"Unknown prioritization_method: {method}")

    return w_s / w_s.sum()


def select_tasks_for_evolution(
    config: DictConfig, archive: TaskArchive, idx: int, k: int
) -> list[str]:
    """Selects frontier tasks for evolution (parent selection).

    A task is on the frontier if:
    1. It has an allowed status (configurable).
    2. It has not exceeded the maximum branching factor.
    3. (Optional) It does not have a viable child that supersedes it.

    Args:
        config: Hydra configuration.
        archive: The task archive.
        idx: Current session index.
        k: Number of tasks to select.

    Returns:
        List of task IDs for evolution.
    """
    # Determine selection strategy
    if config.ablation:
        return _select_tasks_unrestricted(archive, k)

    parent_selection = config.dicode_manager.parent_selection
    if parent_selection == "strict":
        return _select_tasks_frontier(
            archive,
            k,
            parent_statuses={"A", "B"},
            success_statuses={"A", "B", "C"},
            max_children=5,
            sort_by_score=True,
        )
    elif parent_selection == "lenient":
        return _select_tasks_frontier(
            archive,
            k,
            parent_statuses={"A", "B", "C", "D"},
            success_statuses={"A", "B", "C"},
            max_children=5,
            sort_by_score=False,
        )
    else:
        return _select_tasks_unrestricted(archive, k)


def _select_tasks_frontier(
    archive: TaskArchive,
    k: int,
    parent_statuses: set[str],
    success_statuses: set[str],
    max_children: int,
    sort_by_score: bool,
) -> list[str]:
    """Selects frontier tasks with lineage-aware filtering.

    Args:
        archive: The task archive.
        k: Number of tasks to select.
        parent_statuses: Set of allowed parent statuses.
        success_statuses: Child statuses that "graduate" a parent.
        max_children: Maximum children per node.
        sort_by_score: Whether to sort by priority_score before selection.

    Returns:
        List of selected task IDs.
    """
    eligible_tasks = []

    for node_id, data in archive.graph.nodes(data=True):
        status = data.get("status")

        # Filter 1: Allowed status
        if status not in parent_statuses:
            continue

        # Filter 2: Branching cap
        children = list(archive.graph.successors(node_id))
        if len(children) >= max_children:
            continue

        # Filter 3: No viable child
        has_viable_child = any(
            archive.graph.has_node(child_id)
            and archive.graph.nodes[child_id].get("status") in success_statuses
            for child_id in children
        )

        if not has_viable_child:
            eligible_tasks.append(node_id)

    print(f"  Found {len(eligible_tasks)} frontier tasks.")

    # Fallback: if no frontier tasks, use any mastered task
    if not eligible_tasks:
        print("  Warning: No frontier tasks found. Fallback to mastered tasks.")
        eligible_tasks = [
            n for n, d in archive.graph.nodes(data=True) if d.get("status") == "A"
        ]

    # Selection
    if sort_by_score:
        eligible_tasks.sort(
            key=lambda tid: archive.graph.nodes[tid].get("priority_score", 0.0),
            reverse=True,
        )
        selected = eligible_tasks[:k]
    else:
        selected = random.sample(eligible_tasks, min(k, len(eligible_tasks)))

    return _replicate_to_fill(selected, k)


def _select_tasks_unrestricted(archive: TaskArchive, k: int) -> list[str]:
    """Selects tasks without lineage restrictions (for ablation).

    Args:
        archive: The task archive.
        k: Number of tasks to select.

    Returns:
        List of selected task IDs.
    """
    allowed_statuses = {"A", "B", "C", "D"}
    eligible_tasks = [
        node_id
        for node_id, data in archive.graph.nodes(data=True)
        if data.get("status") in allowed_statuses
    ]

    print(f"  Found {len(eligible_tasks)} tasks (unrestricted).")

    if len(eligible_tasks) >= k:
        selected = random.sample(eligible_tasks, k)
    else:
        selected = eligible_tasks[:]

    return _replicate_to_fill(selected, k)


def _replicate_to_fill(selected: list[str], k: int) -> list[str]:
    """Replicates selected tasks to fill batch size k.

    Args:
        selected: List of selected task IDs.
        k: Target batch size.

    Returns:
        List of task IDs with length k (if possible).
    """
    num_selected = len(selected)
    if num_selected == 0:
        return []

    if num_selected >= k:
        return selected

    factor = k // num_selected
    remainder = k % num_selected
    result = selected * factor + selected[:remainder]
    random.shuffle(result)
    return result
