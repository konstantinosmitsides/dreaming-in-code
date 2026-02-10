# --- Third-Party ---
import hydra
import jax
import jax.numpy as jnp
import wandb
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from flax.training.train_state import TrainState
from omegaconf import DictConfig

# --- Local Modules ---
# from evaluation_ import main as evaluate
from dicode.craftax_evaluation import main as evaluate

# from dicode.ppo_rnn import run_evaluation_rollouts
from dicode.ppo_tr import run_evaluation_rollouts
from dicode.task_utils import get_achievement_multi_hot, load_tasks_from_env_codes
from dicode.utils.general.train_state_utils import load_weights_only

from minicraftax.envs.craftax import CraftaxAugObsTrain

# from multitask_evaluation import make_multitask_evaluate
from dicode.dreaming.gen_manager import GenManager, TaskArchive
from dicode.dreaming.llm import LLM


def run_session_evaluation(
	config: DictConfig,
	rng: jax.Array,
	rl_train_state: TrainState,
	gen_manager: GenManager,
	current_session_idx: int,
	global_env_steps: int,
) -> tuple[jax.Array, dict]:
	"""Runs the standard Craftax evaluation against the current agent state."""
	print("--- [Main Thread] Running evaluation on Craftax... ---")

	if config.training.condition_on_task:
		# static_params = StaticEnvParams()
		# env_params = EnvParams()
		# task = Env(static_params, env_params)
		# eval_env = MiniCraftaxTrain(task)
		eval_env = CraftaxAugObsTrain()
		if config.training.conditioning_type == "embedding":
			eval_label = eval_env.label
			INSTRUCTION = "Generate an embedding for this list of achievements capturing the conceptual skills the agent learns if it achieves these achievements."
			eval_embedding_result = gen_manager.selector.embedding_model.get_embedding(
				eval_label, instruction=INSTRUCTION
			)
			eval_embedding_single = jnp.array(eval_embedding_result[0]["embedding"])

			# --- APPLY NORMALIZATION HERE ---
			# 2. Calculate the L2 norm (length) of the vector
			norm = jnp.linalg.norm(eval_embedding_single)

			# 3. Normalize the vector, adding 1e-8 for numerical stability
			normalized_embedding = eval_embedding_single / (norm + 1e-8)

			eval_embedding_replicated = jnp.tile(
				normalized_embedding, (config.evaluation.num_envs, 1)
			)

		else:
			print("Generating one-hot embedding")
			eval_embedding_replicated = jnp.tile(
				get_achievement_multi_hot(eval_env.relevant_achievements),
				(config.evaluation.num_envs, 1),
			)
	else:
		eval_embedding_replicated = None

	rng, eval_rng = jax.random.split(rng)
	evaluation_metrics = evaluate(
		config, eval_rng, train_state=rl_train_state, eval_embedding=eval_embedding_replicated
	)
	# evaluation_metrics = process_evaluation_metrics(eval_metrics_raw)
	print(f"  - Evaluation Metrics: {evaluation_metrics}")

	if config.use_wandb:
		eval_log_data = {"session": current_session_idx, "global_env_steps": global_env_steps}
		for key, value in evaluation_metrics.items():
			eval_log_data[f"evaluation/{key}"] = value
		wandb.log(eval_log_data)

	return rng, evaluation_metrics


def _get_new_task_embeddings(
	config: DictConfig,
	task_classes: list,
	embedding_model: LLM,
) -> jax.Array | None:
	"""Generates embeddings for a list of new task classes."""
	if not config.training.condition_on_task:
		return None

	num_new_tasks = len(task_classes)
	print(f"  - Generating embeddings for {num_new_tasks} new tasks...")
	# Instantiate dummy tasks to get labels
	dummy_static_params = StaticEnvParams()
	dummy_env_params = EnvParams()
	if config.training.conditioning_type == "embedding":
		task_labels = [cls(dummy_static_params, dummy_env_params).label for cls in task_classes]
		INSTRUCTION = "Generate an embedding for this list of achievements capturing the conceptual skills the agent learns if it achieves these achievements."
		embedding_results = embedding_model.get_embedding(task_labels, instruction=INSTRUCTION)

		# 1. Get the raw embedding table (shape: [num_new_tasks, embedding_size])
		raw_embedding_table = jnp.array([res["embedding"] for res in embedding_results])

		# --- APPLY NORMALIZATION HERE ---

		# 2. Calculate the L2 norm for each vector (along axis=1)
		#    keepdims=True makes it shape [num_new_tasks, 1] for broadcasting
		norms = jnp.linalg.norm(raw_embedding_table, axis=1, keepdims=True)

		# 3. Normalize the table, adding 1e-8 for numerical stability
		normalized_table = raw_embedding_table / (norms + 1e-8)

	else:
		print(f"Generating one-hot embeddings for {len(task_classes)} tasks...")
		ach_lists = [cls(None, None).relevant_achievements for cls in task_classes]
		normalized_table = jnp.array(
			[get_achievement_multi_hot(ach_list) for ach_list in ach_lists]
		)

	return normalized_table


def evaluate_new_tasks(
	config: DictConfig,
	rng: jax.Array,
	train_state: TrainState,
	new_task_ids: list[str],
	archive: TaskArchive,
	embedding_model: LLM,
) -> dict:
	"""Evaluates a list of newly generated tasks using the current agent state
	by running rollouts and collecting raw trajectory data for scoring.
	"""
	if not new_task_ids:
		return {}

	print(f"  - Evaluating {len(new_task_ids)} newly generated tasks...")

	# 1. Load Task Classes from Archive Code
	task_classes, _ = load_tasks_from_env_codes(archive, new_task_ids)
	num_new_tasks = len(task_classes)
	if num_new_tasks == 0:
		print("  - Warning: Could not load any task classes for evaluation.")
		return {}

	# 2. Get Embeddings if needed
	task_embeddings = _get_new_task_embeddings(config, task_classes, embedding_model)

	# 3. Create Achievement Mask (needed for the Smart Calculator)
	num_total_achievements = len(Achievement)
	task_achievement_mask = jnp.zeros((num_new_tasks, num_total_achievements), dtype=jnp.bool)
	task_completed_mask = jnp.zeros((num_new_tasks, num_total_achievements), dtype=jnp.bool)
	for i, task_cls in enumerate(task_classes):
		# Instantiate with dummy params to access attributes
		temp_task = task_cls(StaticEnvParams(), EnvParams())
		if temp_task.relevant_achievements:
			achievement_indices = jnp.array([ach.value for ach in temp_task.relevant_achievements])
			task_achievement_mask = task_achievement_mask.at[i, achievement_indices].set(True)
		if temp_task.completed_achievements:
			completed_indices = jnp.array([ach.value for ach in temp_task.completed_achievements])
			task_completed_mask = task_completed_mask.at[i, completed_indices].set(True)

	# 4. Call our new "heavyweight" rollout collector
	# --- START OF REPLACED BLOCK ---
	print("  - Running JIT-compiled evaluation rollouts...")

	# Use the *validation* config for the number of updates
	num_validation_updates = config.validation.rollout_updates

	session_results = run_evaluation_rollouts(
		config,  # Pass global config
		rng,
		task_classes,
		num_training_updates=num_validation_updates,
		train_state=train_state,  # Pass the current agent state
		task_embeddings=task_embeddings,
	)
	print("  - Evaluation run finished.")

	# 5. Extract the raw data needed for the calculator
	scoring_window_data = session_results.get("metrics", {}).get("scoring_window_data")

	if scoring_window_data is None:
		print("  - Warning: Evaluation rollouts produced no scoring_window_data.")
		return {}
	# --- END OF REPLACED BLOCK ---

	# 6. Return the raw data and the mask
	return {
		"scoring_window_data": scoring_window_data,
		"task_achievement_mask": task_achievement_mask,
		"task_completed_mask": task_completed_mask,
	}


def process_evaluation_metrics(raw_metrics: dict) -> dict:
	"""Processes the raw metrics from the evaluation script into a clean,
	standardized dictionary for the GenManager engine.

	Args:
	    raw_metrics: The dictionary returned by the evaluation script.

	Returns:
	    A clean dictionary, e.g.,
	    {"mean_performance": 3.52, "skill_collect_wood": 99.89, ...}

	"""
	processed_metrics = {}

	total_reward_achieved = 0.0
	total_possible_reward = 0.0

	for key, value in raw_metrics.items():
		if key == "returned_episode_lengths":
			processed_metrics["average_episode_length"] = float(value)
			continue
		if key.startswith("Achievements/"):
			# exctract the skill name
			skill_name_raw = key.split("/")[-1]

			try:
				# Convert string name tp enum
				achievement_enum = Achievement[skill_name_raw.upper()]
				achievement_index = achievement_enum.value

				# Get the reward for this achievement
				reward_value = ACHIEVEMENT_REWARD_MAP[achievement_index]

				# Add to total possible reward
				total_possible_reward += reward_value

				# Add the achieved portion of the reward
				# value is 0-100 so convert 0-1
				achieved_proportion = value / 100.0
				total_reward_achieved += reward_value * achieved_proportion

			except:
				pass

			clean_key = f"skill_{skill_name_raw}"

			processed_metrics[clean_key] = float(value)

	if total_possible_reward > 0:
		mean_reward_percentage = (total_reward_achieved / total_possible_reward) * 100.0
	else:
		mean_reward_percentage = 0.0

	processed_metrics["mean_performance"] = raw_metrics["returned_episode_returns"] / 226.0 * 100.0

	processed_metrics["mean_return"] = raw_metrics["returned_episode_returns"]

	return processed_metrics


@hydra.main(
	version_base="1.2",
	config_path="/home_nfs/konstantinos/projects/MiniCraftax/conf/",
	config_name="config",
)
def main(config: DictConfig) -> None:
	gen_manager = GenManager(config)
	dummy_env = CraftaxAugObsTrain(
		condition_on_task=config.training.condition_on_task,
		conditioning_type=config.training.conditioning_type,
		embedding_size=config.gen_manager.embedding_model.embedding_size,
		task_embeddings=jnp.zeros((1, config.gen_manager.embedding_model.embedding_size)),
	)
	rl_train_state = load_weights_only(
		checkpoint_path=config.rl_ckpt_path,
		env=dummy_env,
		env_params=dummy_env.default_params,
		config=config.training,
	)

	_, evaluation_metrics = run_session_evaluation(
		config,
		jax.random.PRNGKey(config.seed),
		rl_train_state,
		gen_manager,
		0,
		0,
	)
	print(evaluation_metrics)


if __name__ == "__main__":
	main()
