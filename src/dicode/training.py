"""Training session utilities for DiCode.

This module provides functions for:
1. Running training sessions on batches of tasks.
2. Processing training metrics from the PPO agent.
3. Extracting evaluation metrics for the original task.
"""

# --- Third-Party ---
import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from flax.training.train_state import TrainState
from omegaconf import DictConfig

# --- Local Modules ---
from dicode.dreaming.gen_manager import GenManager
from dicode.ppo_tr import run_training_session
from dicode.scoring import calculate_scores_from_snapshot
from dicode.task_utils import (
    categorize_active_tasks,
    get_achievement_multi_hot,
    load_tasks_from_env_codes,
    update_archive_statuses,
)
from minicraftax.tasks.seed_tasks.original import Env as OriginalTask


# Embedding instruction (shared across the codebase)
EMBEDDING_INSTRUCTION = (
    "Generate an embedding for this list of achievements capturing "
    "the conceptual skills the agent learns if it achieves these achievements."
)


def run_session_training(
    config: DictConfig,
    rng: jax.Array,
    rl_train_state: TrainState,
    gen_manager: GenManager,
    global_update_step: int,
    global_env_steps: int,
    current_session_idx: int,
    sampled_task_ids: list[str],
    original_return_prev_session: float = 0.0,
) -> tuple:
    """Runs a full training session on a sampled batch of tasks.

    This function handles:
    1. Loading task classes and generating embeddings.
    2. Running the PPO training loop.
    3. Processing metrics and updating the task archive.
    4. Optionally resetting the optimizer state.

    Args:
        config: Hydra configuration.
        rng: JAX PRNG key.
        rl_train_state: Current agent training state.
        gen_manager: The GenManager instance.
        global_update_step: Current global update step.
        global_env_steps: Current global environment steps.
        current_session_idx: Current session index.
        sampled_task_ids: List of task IDs to train on.
        original_return_prev_session: Previous session's original task return.

    Returns:
        Tuple of (rng, rl_train_state, global_update_step, global_env_steps,
                  training_metrics, num_updates_in_session, categorized_tasks,
                  evaluation_metrics).
    """
    print("--- [Main Thread] Starting training session... ---")
    print(f"Training on {len(sampled_task_ids)} sampled tasks: {sampled_task_ids}")

    # Load task classes
    sampled_task_classes, successful_sampled_ids = load_tasks_from_env_codes(
        gen_manager.archive, sampled_task_ids
    )

    if not sampled_task_classes:
        print("  Error: Could not load any task classes. Skipping training.")
        return (rng, rl_train_state, global_update_step, global_env_steps, {}, 0, {}, {})

    # Add original task for evaluation
    all_task_classes = sampled_task_classes + [OriginalTask]
    all_task_ids = successful_sampled_ids + ["original_craftax"]
    num_tasks_in_session = len(all_task_classes)

    # Create achievement masks
    task_achievement_mask, task_completed_mask = _create_achievement_masks(
        all_task_classes
    )

    # Generate embeddings if needed
    normalized_table = _generate_embeddings_for_session(
        config, all_task_classes, gen_manager
    )

    # Calculate task distribution
    task_distribution_proportions = _calculate_task_distribution(
        config, len(sampled_task_classes)
    )

    # Run training
    print("  Starting training run...")
    rng, train_rng = jax.random.split(rng)

    if rl_train_state is None:
        raise RuntimeError("rl_train_state is None. Should have been initialized.")

    session_results = run_training_session(
        config,
        train_rng,
        all_task_classes,
        num_training_updates=config.dicode_manager.max_updates_per_session,
        train_state=rl_train_state,
        task_embeddings=normalized_table,
        task_distribution_proportions=task_distribution_proportions,
        global_update_step=global_update_step,
        current_original_return=original_return_prev_session,
    )
    print("  Training run finished.")

    # Update agent state
    rl_train_state = session_results["train_state"]

    if config.dicode_manager.reset_opt_state:
        rl_train_state = _reset_optimizer_state(config, rl_train_state)

    # Update global counters
    num_updates_in_session = int(session_results["metrics"]["num_updates_done"])
    num_env_steps_in_session = int(session_results["metrics"]["num_env_steps_done"])
    global_update_step += num_updates_in_session
    global_env_steps += num_env_steps_in_session

    print(f"  Session: {num_updates_in_session} updates, {num_env_steps_in_session} env steps")
    print(f"  Global: {global_update_step} updates, {global_env_steps} env steps")

    # Process metrics
    training_metrics = {}
    categorized_tasks = {}
    evaluation_metrics = {}

    if num_updates_in_session > 0:
        print("  Processing training metrics...")
        training_metrics = process_training_metrics(
            all_task_ids,
            session_results["metrics"],
            num_tasks_in_session,
            task_achievement_mask,
            task_completed_mask,
            config,
            force_include_achievements_indices=[num_tasks_in_session - 1],
        )

        # Extract original task metrics
        evaluation_metrics = extract_and_format_original_metrics(
            training_metrics, "original_craftax"
        )
        print(f"  Evaluation metrics: {evaluation_metrics}")

        # Remove original task from training metrics
        if "original_craftax" in training_metrics:
            del training_metrics["original_craftax"]

        # Update archive
        _update_archive_with_metrics(
            gen_manager, training_metrics, current_session_idx, config
        )

        # Categorize tasks
        categorized_tasks = categorize_active_tasks(training_metrics)
        update_archive_statuses(gen_manager.archive, categorized_tasks)
    else:
        print("  Skipping metric update (no training occurred).")

    return (
        rng,
        rl_train_state,
        global_update_step,
        global_env_steps,
        training_metrics,
        num_updates_in_session,
        categorized_tasks,
        evaluation_metrics,
    )


def extract_and_format_original_metrics(
    training_metrics: dict, original_task_id: str
) -> dict:
    """Extracts and formats metrics for the original task.

    Args:
        training_metrics: Dictionary of per-task training metrics.
        original_task_id: The ID of the original task.

    Returns:
        Dictionary of formatted evaluation metrics.
    """
    if original_task_id not in training_metrics:
        print(f"Warning: Original task {original_task_id} not found in metrics.")
        return {}

    metrics = training_metrics[original_task_id]
    ach_srs = metrics.get("achievement_srs", {})
    processed = {}

    # Process achievement success rates
    for ach_name, value in ach_srs.items():
        processed[f"skill_{ach_name}"] = float(value)

    # Add aggregate metrics
    if "mean_return_percentage" in metrics:
        processed["mean_performance"] = metrics["mean_return_percentage"]
    if "mean_return" in metrics:
        processed["mean_return"] = metrics["mean_return"]
    if "average_episode_length" in metrics:
        processed["average_episode_length"] = metrics["average_episode_length"]

    return processed


def process_training_metrics(
    task_ids: list[str],
    session_metrics: dict,
    num_tasks: int,
    task_achievement_mask: np.ndarray,
    task_completed_mask: np.ndarray,
    config: DictConfig,
    force_include_achievements_indices: list[int] | None = None,
) -> dict:
    """Processes raw training metrics using the scoring module.

    Args:
        task_ids: List of task IDs.
        session_metrics: Raw metrics from the training session.
        num_tasks: Number of tasks in the session.
        task_achievement_mask: Boolean mask of relevant achievements per task.
        task_completed_mask: Boolean mask of completed achievements per task.
        config: Hydra configuration.
        force_include_achievements_indices: Task indices that should always
            include achievement metrics (e.g., original task).

    Returns:
        Dictionary mapping task_id -> metrics.
    """
    scoring_window_data = session_metrics.get("scoring_window_data")

    if scoring_window_data is None:
        print("  [Scoring] Error: 'scoring_window_data' not found. Returning empty.")
        return {}

    # Call the scoring module
    scores_by_idx = calculate_scores_from_snapshot(
        scoring_window_data,
        num_tasks,
        task_achievement_mask,
        task_completed_mask,
        config,
        force_include_achievements_indices,
    )

    # Map integer indices back to task IDs
    final_metrics = {}
    if len(task_ids) != len(scores_by_idx):
        print(f"  [Scoring] Warning: Mismatch in task_ids ({len(task_ids)}) "
              f"and scored results ({len(scores_by_idx)}).")

    for i, task_id in enumerate(task_ids):
        task_idx_str = str(i)
        if task_idx_str in scores_by_idx:
            final_metrics[task_id] = scores_by_idx[task_idx_str]
        else:
            print(f"  [Scoring] Warning: No score for {task_id} (index {i}).")

    return final_metrics


def _create_achievement_masks(
    task_classes: list,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Creates achievement masks for task classes."""
    num_tasks = len(task_classes)
    num_achievements = len(Achievement)

    task_achievement_mask = jnp.zeros((num_tasks, num_achievements), dtype=jnp.bool_)
    task_completed_mask = jnp.zeros((num_tasks, num_achievements), dtype=jnp.bool_)

    for i, task_cls in enumerate(task_classes):
        temp_task = task_cls(StaticEnvParams(), EnvParams())
        if temp_task.relevant_achievements:
            indices = jnp.array([ach.value for ach in temp_task.relevant_achievements])
            task_achievement_mask = task_achievement_mask.at[i, indices].set(True)
        if temp_task.completed_achievements:
            indices = jnp.array([ach.value for ach in temp_task.completed_achievements])
            task_completed_mask = task_completed_mask.at[i, indices].set(True)

    return task_achievement_mask, task_completed_mask


def _generate_embeddings_for_session(
    config: DictConfig, task_classes: list, gen_manager: GenManager
) -> jnp.ndarray | None:
    """Generates normalized embeddings for session tasks."""
    if not config.training.condition_on_task:
        return None

    if config.training.condition_on_task == "embedding":
        print(f"  Generating embeddings for {len(task_classes)} tasks...")
        task_labels = [cls(None, None).label for cls in task_classes]
        embedding_results = gen_manager.selector.embedding_model.get_embedding(
            task_labels, instruction=EMBEDDING_INSTRUCTION
        )
        raw_table = jnp.array([res["embedding"] for res in embedding_results])

        # L2 normalize
        norms = jnp.linalg.norm(raw_table, axis=1, keepdims=True)
        return raw_table / (norms + 1e-8)
    else:
        print(f"Generating one-hot embeddings for {len(task_classes)} tasks...")
        ach_lists = [cls(None, None).relevant_achievements for cls in task_classes]
        return jnp.array([get_achievement_multi_hot(ach_list) for ach_list in ach_lists])


def _calculate_task_distribution(
    config: DictConfig, num_curriculum_tasks: int
) -> jnp.ndarray:
    """Calculates task sampling distribution."""
    original_proportion = config.dicode_manager.get("original_task_proportion", 0.2)

    if num_curriculum_tasks > 0:
        other_proportion = (1.0 - original_proportion) / num_curriculum_tasks
        proportions = jnp.concatenate([
            jnp.full(num_curriculum_tasks, other_proportion),
            jnp.array([original_proportion]),
        ])
    else:
        proportions = jnp.array([1.0])

    return proportions / jnp.sum(proportions)


def _reset_optimizer_state(
    config: DictConfig, train_state: TrainState
) -> TrainState:
    """Resets optimizer state for the next session."""
    print("Resetting optimizer state for next session...")

    if config.training.anneal_lr:
        lr_schedule = optax.exponential_decay(
            init_value=config.training.lr,
            transition_steps=1,
            decay_rate=config.training.lr_decay_rate,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config.training.max_grad_norm),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config.training.max_grad_norm),
            optax.adam(config.training.lr, eps=1e-5),
        )

    return TrainState.create(
        apply_fn=train_state.apply_fn,
        params=train_state.params,
        tx=tx,
    )


def _update_archive_with_metrics(
    gen_manager: GenManager,
    training_metrics: dict,
    current_session_idx: int,
    config: DictConfig,
) -> None:
    """Updates the archive with training metrics."""
    updated_count = 0

    for task_id, metrics in training_metrics.items():
        priority_score = metrics.get("priority_score", 0.0)
        gen_manager.archive.update_node_priority_score(task_id, priority_score)

        if config.dicode_manager.score_function == "learnability":
            gen_manager.archive.update_node_learnability(task_id, priority_score)

        gen_manager.archive.update_node_performance(
            current_session_idx, {task_id: metrics}
        )
        gen_manager.archive.update_node_session_last_trained(task_id, current_session_idx)
        updated_count += 1

    print(f"  Updated priority_score for {updated_count} tasks.")
