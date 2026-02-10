"""Experiment setup utilities for DiCode.

This module provides functions for:
1. Initializing W&B, GenManager, and checkpoint managers.
2. Running the initial seed training phase.
"""

# --- Standard Library ---
import os
import time

# --- Third-Party ---
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

# --- Local Modules ---
from dicode.dreaming.gen_manager import GenManager
from dicode.evolution import attempt_to_activate_task
from dicode.logging_utils import log_wandb_table_append
from dicode.ppo_tr import run_training_session
from dicode.task_utils import (
    categorize_active_tasks,
    get_achievement_multi_hot,
    load_tasks_from_env_codes,
    update_archive_statuses,
)
from dicode.training import extract_and_format_original_metrics, process_training_metrics
from dicode.utils.general.train_state_utils import load_weights_only
from minicraftax.envs.craftax import CraftaxAugObsTrain
from minicraftax.tasks.seed_tasks.original import Env as OriginalTask


def setup_experiment(config: DictConfig) -> tuple:
    """Initializes the entire experiment setup.

    This includes W&B, GenManager, RNG keys, checkpoint managers,
    and loading any existing agent state.

    Args:
        config: Hydra configuration object.

    Returns:
        Tuple of (rng, gen_manager, rl_ckpt_manager, rl_train_state,
                  global_update_step, global_env_steps, latest_step,
                  cumulative_compiled, cumulative_activated).
    """
    if config.use_wandb:
        _setup_wandb(config)

    print("--- Initializing Online Curriculum Manager ---")

    # Initialize GenManager and RNG
    gen_manager = GenManager(config)
    rng = jax.random.PRNGKey(config.seed)

    # Set up checkpoint manager
    rl_ckpt_path = os.path.join(os.getcwd(), config.checkpoint_dir)
    orbax_checkpointer = PyTreeCheckpointer()
    ckpt_options = CheckpointManagerOptions(
        max_to_keep=config.training.max_checkpoints_to_keep,
        keep_period=config.training.checkpoint_keep_period,
        create=True,
    )
    rl_ckpt_manager = CheckpointManager(
        rl_ckpt_path, orbax_checkpointer, options=ckpt_options
    )

    # Load or initialize agent state
    latest_step = rl_ckpt_manager.latest_step()
    if latest_step is not None:
        print(f"Restoring RL agent state from checkpoint at step {latest_step}...")
        rl_train_state = _load_agent_state(config, rl_ckpt_path)
        print("Successfully loaded weights.")
    else:
        print("No RL agent checkpoint found. Starting from scratch.")
        rl_train_state = None

    # Initialize counters
    global_update_step = 0
    global_env_steps = 0
    cumulative_compiled = 0
    cumulative_activated = 0

    # If resumed, fetch last known steps from W&B
    if config.use_wandb and wandb.run.resumed:
        print("Run resumed. Fetching last logged steps from W&B...")
        global_update_step = int(wandb.run.summary.get("global_update_step", 0))
        global_env_steps = int(wandb.run.summary.get("global_env_steps", 0))
        cumulative_compiled = int(
            wandb.run.summary.get("curriculum/num_tasks_compiled_cumulative", 0)
        )
        cumulative_activated = int(
            wandb.run.summary.get("curriculum/num_tasks_activated_cumulative", 0)
        )
        print(f"Resuming from step {global_update_step}, env_steps {global_env_steps}")

    return (
        rng,
        gen_manager,
        rl_ckpt_manager,
        rl_train_state,
        global_update_step,
        global_env_steps,
        latest_step,
        cumulative_compiled,
        cumulative_activated,
    )


def _setup_wandb(config: DictConfig) -> None:
    """Initializes Weights & Biases logging."""
    config_dict = OmegaConf.to_container(config, resolve=True)
    run_id = os.getenv("WANDB_RUN_ID", wandb.util.generate_id())

    # Check for API key (safeguard against crashing if missing)
    if "WANDB_API_KEY" not in os.environ:
        print("\n" + "!" * 80)
        print("WARNING: WANDB_API_KEY not found in environment variables.")
        print("         Disabling WandB logging to prevent crashing.")
        print("         To use WandB, add WANDB_API_KEY to your .env file or export it.")
        print("!" * 80 + "\n")
        config.use_wandb = False
        return

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config_dict,
        name=f"DiCode-run-{int(time.time())}",
        save_code=True,
        id=run_id,
        resume="allow",
    )

    # Define custom x-axes
    wandb.define_metric("session")
    wandb.define_metric("global_update_step")
    wandb.define_metric("global_env_steps")
    wandb.define_metric("curriculum/*", step_metric="session")
    wandb.define_metric("training/*", step_metric="session")
    wandb.define_metric("evaluation/*", step_metric="global_env_steps")


def _load_agent_state(config: DictConfig, rl_ckpt_path: str) -> TrainState:
    """Loads agent weights from checkpoint."""
    dummy_env = CraftaxAugObsTrain(
        condition_on_task=config.training.condition_on_task,
        conditioning_type=config.training.conditioning_type,
        embedding_size=config.gen_manager.embedding_model.embedding_size,
        task_embeddings=jnp.zeros((1, config.gen_manager.embedding_model.embedding_size)),
    )
    return load_weights_only(
        checkpoint_path=rl_ckpt_path,
        env=dummy_env,
        env_params=dummy_env.default_params,
        config=config.training,
        load_opt_state=True,
    )


def run_initial_seed_training(
    config: DictConfig,
    gen_manager: GenManager,
    rng: jax.Array,
    rl_train_state: TrainState | None,
    global_update_step: int,
    global_env_steps: int,
) -> tuple:
    """Runs the initial training phase on seed tasks.

    This bootstraps the agent and calculates initial learnability for seeds.

    Args:
        config: Hydra configuration.
        gen_manager: The GenManager instance.
        rng: JAX PRNG key.
        rl_train_state: Current agent state (None for fresh start).
        global_update_step: Current global update step.
        global_env_steps: Current global environment steps.

    Returns:
        Tuple of (rng, rl_train_state, global_update_step, global_env_steps,
                  evaluation_metrics).
    """
    print(f"\n{'=' * 60}")
    print("--- Starting Initial Seed Training Phase ---")
    print(f"{'=' * 60}")

    # Get seed tasks
    seed_task_ids = [
        node_id
        for node_id, data in gen_manager.archive.graph.nodes(data=True)
        if data.get("status") == "seed"
    ]
    if not seed_task_ids:
        raise ValueError("No seed tasks found in the archive.")
    print(f"Found {len(seed_task_ids)} seed tasks: {seed_task_ids}")

    seed_task_classes, successful_seed_ids = load_tasks_from_env_codes(
        gen_manager.archive, seed_task_ids
    )

    # Add original task
    all_seed_task_classes = seed_task_classes + [OriginalTask]
    all_seed_task_ids = successful_seed_ids + ["original_craftax"]
    num_seed_tasks = len(all_seed_task_classes)

    # Create achievement masks
    task_achievement_mask, task_completed_mask = _create_achievement_masks(
        all_seed_task_classes
    )

    # Generate embeddings if needed
    normalized_table = _generate_task_embeddings(
        config, all_seed_task_classes, gen_manager
    )

    # Calculate task distribution
    task_distribution_proportions = _calculate_task_distribution(
        config, len(seed_task_classes)
    )

    # Train on seeds
    print("Starting training session on seed tasks...")
    rng, seed_train_rng = jax.random.split(rng)
    seed_session_results = run_training_session(
        config,
        seed_train_rng,
        all_seed_task_classes,
        num_training_updates=config.dicode_manager.max_updates_per_session * 2,
        train_state=None,
        task_embeddings=normalized_table,
        task_distribution_proportions=task_distribution_proportions,
        global_update_step=global_update_step,
    )
    rl_train_state = seed_session_results["train_state"]
    print("Seed training finished.")

    # Update global counters
    num_seed_updates = int(seed_session_results["metrics"]["num_updates_done"])
    num_seed_env_steps = seed_session_results["metrics"]["num_env_steps_done"]
    global_update_step += num_seed_updates
    global_env_steps += num_seed_env_steps

    # Process metrics
    print("Processing seed metrics...")
    seed_training_metrics = process_training_metrics(
        all_seed_task_ids,
        seed_session_results["metrics"],
        num_seed_tasks,
        task_achievement_mask,
        task_completed_mask,
        config,
        force_include_achievements_indices=[num_seed_tasks - 1],
    )

    # Extract original task metrics
    evaluation_metrics = extract_and_format_original_metrics(
        seed_training_metrics, "original_craftax"
    )
    print(f"  Extracted evaluation metrics: {evaluation_metrics}")

    # Remove original task from seed metrics (it's not a curriculum task)
    if "original_craftax" in seed_training_metrics:
        del seed_training_metrics["original_craftax"]

    # Update archive with seed scores
    _update_seed_scores(gen_manager, seed_training_metrics, config)

    # Categorize and update statuses
    seed_categorized_tasks = categorize_active_tasks(seed_training_metrics)
    update_archive_statuses(gen_manager.archive, seed_categorized_tasks)
    gen_manager.archive.save_graph()

    # Log to W&B
    if config.use_wandb:
        _log_seed_phase(
            config,
            global_update_step,
            global_env_steps,
            seed_training_metrics,
            seed_categorized_tasks,
        )

    print("--- Finished Initial Seed Training Phase ---")
    return rng, rl_train_state, global_update_step, global_env_steps, evaluation_metrics


def _create_achievement_masks(
    task_classes: list,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Creates achievement masks for a list of task classes."""
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


def _generate_task_embeddings(
    config: DictConfig, task_classes: list, gen_manager: GenManager
) -> jnp.ndarray | None:
    """Generates normalized embeddings for task classes."""
    if not config.training.condition_on_task:
        return None

    if config.training.conditioning_type == "embedding":
        print(f"Generating embeddings for {len(task_classes)} tasks...")
        task_labels = [cls(None, None).label for cls in task_classes]
        instruction = (
            "Generate an embedding for this list of achievements capturing "
            "the conceptual skills the agent learns if it achieves these achievements."
        )
        embedding_results = gen_manager.selector.embedding_model.get_embedding(
            task_labels, instruction=instruction
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
    """Calculates task sampling distribution with original task proportion."""
    original_task_proportion = config.dicode_manager.get("original_task_proportion", 0.2)

    if num_curriculum_tasks > 0:
        other_proportion = (1.0 - original_task_proportion) / num_curriculum_tasks
        proportions = jnp.concatenate([
            jnp.full(num_curriculum_tasks, other_proportion),
            jnp.array([original_task_proportion]),
        ])
    else:
        proportions = jnp.array([1.0])

    return proportions / jnp.sum(proportions)


def _update_seed_scores(
    gen_manager: GenManager, seed_training_metrics: dict, config: DictConfig
) -> None:
    """Updates archive with seed task scores."""
    print("Calculating and storing scores for seeds...")

    for task_id, metrics in seed_training_metrics.items():
        priority_score = metrics.get("priority_score", 0.0)
        gen_manager.archive.update_node_priority_score(task_id, priority_score)

        if config.dicode_manager.score_function == "learnability":
            gen_manager.archive.update_node_learnability(task_id, priority_score)
        else:
            sr = metrics.get("sr", -1.0)
            clipped_sr = np.clip(sr, 0.0, 1.0) if sr >= 0 else 0.0
            learnability = clipped_sr * (1.0 - clipped_sr)
            gen_manager.archive.update_node_learnability(task_id, learnability)

        gen_manager.archive.update_node_performance(0, {task_id: metrics})
        gen_manager.archive.update_node_session_last_trained(task_id, 0)
        attempt_to_activate_task(gen_manager, task_id, priority_score, config)

        print(f"  {task_id}: SR={metrics.get('sr', -1.0):.3f}, "
              f"Priority={priority_score:.3f}")


def _log_seed_phase(
    config: DictConfig,
    global_update_step: int,
    global_env_steps: int,
    seed_training_metrics: dict,
    seed_categorized_tasks: dict,
) -> None:
    """Logs seed phase results to W&B."""
    seed_log_data = {
        "global_update_step": global_update_step,
        "global_env_steps": global_env_steps,
        "session": 0,
    }

    # Build status map
    status_map = {}
    for status in ["A", "B", "C", "D"]:
        for task_id in seed_categorized_tasks.get(status, []):
            status_map[task_id] = status

    # Create performance table rows
    columns = ["session", "task_id", "status", "sr", "lp"]
    new_rows = []
    for task_id, metrics in seed_training_metrics.items():
        lp_value = metrics.get("lp", np.nan)
        if np.isnan(lp_value):
            lp_value = None
        new_rows.append([
            0,
            task_id,
            status_map.get(task_id, "N/A"),
            metrics.get("sr", -1.0),
            lp_value,
        ])

    performance_table = log_wandb_table_append(
        table_key="curriculum/task_performance",
        columns=columns,
        new_data_rows=new_rows,
    )
    if performance_table:
        seed_log_data["curriculum/task_performance"] = performance_table

    wandb.log(seed_log_data)
