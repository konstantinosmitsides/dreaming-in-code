"""Load one checkpoint and evaluate it on the held-out Craftax test set.

Used by `experiments/evaluation/run_paper_evaluation.py` for every method in the paper.

Note on task conditioning: DiCode checkpoints carry a task-conditioned observation. All of the
paper's runs used `conditioning_type: one_hot`, which is computed locally from the environment's
achievement list and needs no external services. The `embedding` branch below is retained for
completeness but was not used for any published number -- it would require a running embedding
server.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import json
from omegaconf import DictConfig, OmegaConf
from dicode.task_utils import get_achievement_multi_hot
from dicode.paper_evaluation.craftax_tr_evaluation import main as evaluate

from dicode.utils.general.train_state_utils import load_weights_only

from minicraftax.envs.craftax import CraftaxAugObsTrain  # For dummy env
from dicode.dreaming.gen_manager import GenManager


def evaluate_checkpoint(
    method: str,
	checkpoint_path: str,
	num_envs: int,
	num_steps: int,
	rng: jax.Array,
	config: DictConfig,
):
	"""Evaluates a checkpoint on Craftax.

	Args:
	    checkpoint_path: Path to the checkpoint directory or specific step.
	    task_names: List of task names (corresponding to files in seed_tasks).
	    config: Hydra configuration object.

	Returns:
	    pd.DataFrame: DataFrame containing results for each task.

	"""
	print("--- Starting Evaluation ---")
	print(f"Checkpoint: {checkpoint_path}")

	# 1. Load Train State
	print("\n[1/4] Loading Checkpoint...")
	# We need a dummy env to initialize the network architecture
	eval_embedding_replicated = None

	if method.lower() == "dicode" or method.lower().startswith("ablation"):
		eval_env = CraftaxAugObsTrain()
		if config.training.conditioning_type == "embedding":
			gen_manager = GenManager(config)
			embedding_model = gen_manager.selector.embedding_model
			eval_label = eval_env.label
			INSTRUCTION = "Generate an embedding for this list of achievements capturing the conceptual skills the agent learns if it achieves these achievements."
			eval_embedding_result = embedding_model.get_embedding(
				eval_label, instruction=INSTRUCTION
			)
			eval_embedding_single = jnp.array(eval_embedding_result[0]["embedding"])

			# --- APPLY NORMALIZATION HERE ---
			# 2. Calculate the L2 norm (length) of the vector
			norm = jnp.linalg.norm(eval_embedding_single)

			# 3. Normalize the vector, adding 1e-8 for numerical stability
			normalized_embedding = eval_embedding_single / (norm + 1e-8)
			eval_embedding_replicated = jnp.tile(normalized_embedding, (1, 1))
			embedding_size = eval_embedding_replicated.shape[1]

		else:
			ach_list = eval_env.relevant_achievements
			embedding = get_achievement_multi_hot(ach_list)
			eval_embedding_replicated = jnp.tile(embedding, (1, 1))
			embedding_size = eval_embedding_replicated.shape[1]

		# 1) Load environment
		dummy_env = CraftaxAugObsTrain(
			condition_on_task=config.training.condition_on_task,
			conditioning_type=config.training.conditioning_type,
			embedding_size=embedding_size,
			task_embeddings=eval_embedding_replicated,
		)
		print(f"\n[DEBUG] Config condition_on_task: {config.training.condition_on_task}")
		print(f"[DEBUG] Config conditioning_type: {config.training.conditioning_type}")
		print(f"[DEBUG] Config embedding_size: {embedding_size}")
		print(
			f"[DEBUG] Dummy Env Obs Shape: {dummy_env.observation_space(dummy_env.default_params).shape}"
		)

	else:
		dummy_env = CraftaxAugObsTrain()

	rl_train_state = load_weights_only(
		checkpoint_path=checkpoint_path,
		env=dummy_env,
		env_params=dummy_env.default_params,
		config=config.training,
	)
	print("Checkpoint loaded successfully.")


	if method.lower() == "dicode" or method.lower().startswith("ablation"):
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

			eval_embedding = jnp.tile(
				normalized_embedding, (config.evaluation.num_envs, 1)
			)

		else:
			print("Generating one-hot embedding")
			eval_embedding = jnp.tile(
				get_achievement_multi_hot(eval_env.relevant_achievements),
				(config.evaluation.num_envs, 1),
			)
	else:
		eval_embedding = None


	# 4. Run Evaluation Loop
	print("\n[4/4] Running Evaluation Rollouts...")

	metrics = evaluate(
		config=config,
		rng=rng,
		num_envs=num_envs,
		num_steps=num_steps,
		train_state=rl_train_state,
		eval_embedding=eval_embedding,
	)

	print(f"Returns: {metrics["returns"]}")
	print(f"Mean Return: {metrics["mean_return"]}")
	print(f"Median Return: {metrics["median_return"]}")
	print(f"Max Return :{metrics["max_return"]}")
	print(f"Min Return: {metrics["min_return"]}")
	print(f"Mean Length: {metrics["mean_length"]}")
	print(f"Median Length: {metrics["median_length"]}")
	print(f"Max Length :{metrics["max_length"]}")
	print(f"Min Length: {metrics["min_length"]}")

	print(f"Correlation (Length/Return): {metrics["correlation_length_return"]}")

	print(f"Completion Rate: {metrics["completion_rate"]}")


	return metrics


def convert_to_serializable(obj):
    """
    Recursively converts JAX arrays, NumPy arrays, and Pandas objects 
    into standard Python types (lists, dicts, floats, ints) for JSON serialization.
    """
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (jnp.number, np.number)):
        return obj.item()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()  # or obj.values.tolist() if you prefer pure lists
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    # Handle the OmegaConf config object if it accidentally gets passed in
    elif hasattr(obj, '__dict__'): 
         return str(obj)
    return obj


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Evaluate a single checkpoint on the held-out Craftax test set."
	)
	parser.add_argument("--checkpoint", required=True,
	                    help="Path to one checkpoint directory, e.g. <run>/rl_checkpoints/15400")
	parser.add_argument("--config", required=True,
	                    help="Hydra config.yaml recorded for that run, e.g. <run>/.hydra/config.yaml")
	parser.add_argument("--method", default="DiCode",
	                    help="Method label. Anything starting with 'dicode'/'ablation' takes the "
	                         "task-conditioned observation path.")
	parser.add_argument("--num-envs", type=int, default=1024)
	parser.add_argument("--num-steps", type=int, default=8192)
	parser.add_argument("--output", default=None, help="Optional path to write metrics as JSON.")
	args = parser.parse_args()

	config = OmegaConf.load(args.config)

	start_time = time.time()
	metrics = evaluate_checkpoint(
		method=args.method,
		checkpoint_path=args.checkpoint,
		num_envs=args.num_envs,
		num_steps=args.num_steps,
		rng=jax.random.PRNGKey(0),
		config=config,
	)
	print(f"Evaluated in {time.time() - start_time:.1f}s")
	print(f"mean_return: {metrics['mean_return']}")

	if args.output:
		with open(args.output, "w") as f:
			json.dump(convert_to_serializable(dict(metrics)), f, indent=4)
		print(f"Wrote {args.output}")
