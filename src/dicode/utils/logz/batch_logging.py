import time

import jax.numpy as jnp
import numpy as np
import wandb

batch_logs = {}
log_times = []


def create_log_cl_dict(metric, config):
	"""Creates a flat dictionary for wandb logging from the per-task metrics."""
	to_log = {}

	# Log the global average return if it exists
	if "global_mean_return" in metric:
		to_log["global_mean_return"] = metric["global_mean_return"]
	if "active_tasks_mean_performance" in metric:
		to_log["active_tasks_mean_performance"] = metric["active_tasks_mean_performance"]

	# --- NEW: Unpack the per-task metrics ---
	# This loop creates a separate log entry for each task's performance.
	if "per_task_returns" in metric:
		for i, task_return in enumerate(metric["per_task_returns"]):
			# Only log tasks that have been attempted (return is not 0)
			# This keeps the dashboard clean at the start of training.
			if task_return > 0 or metric["per_task_lengths"][i] > 0:
				# if True:
				to_log[f"task_{i}/mean_return"] = task_return
				to_log[f"task_{i}/mean_length"] = metric["per_task_lengths"][i]
				to_log[f"task_{i}/success_rate"] = metric["per_task_success_rate"][i]

	return to_log


def create_log_dict(info, config):
	to_log = {
		"episode_return": info["returned_episode_returns"],
		"mean_performance": info["returned_episode_returns"] / 226.0 * 100.0,
		"episode_length": info["returned_episode_lengths"],
	}

	sum_achievements = 0
	for k, v in info.items():
		if "achievements" in k.lower():
			to_log[k] = v
			sum_achievements += v / 100.0

	to_log["achievements"] = sum_achievements

	if config.get("TRAIN_ICM") or config.get("USE_RND"):
		to_log["intrinsic_reward"] = info["reward_i"]
		to_log["extrinsic_reward"] = info["reward_e"]

		if config.get("TRAIN_ICM"):
			to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
			to_log["icm_forward_loss"] = info["icm_forward_loss"]
		elif config.get("USE_RND"):
			to_log["rnd_loss"] = info["rnd_loss"]

	return to_log


def batch_log(update_step, log, config, global_env_steps):
	update_step = int(update_step)
	if update_step not in batch_logs:
		batch_logs[update_step] = []

	batch_logs[update_step].append(log)

	if len(batch_logs[update_step]) == config.num_repeats:
		agg_logs = {}
		for key in batch_logs[update_step][0]:
			agg = []
			if key in ["goal_heatmap"]:
				agg = [batch_logs[update_step][0][key]]
			else:
				for i in range(config.num_repeats):
					val = batch_logs[update_step][i][key]
					if not jnp.isnan(val):
						agg.append(val)

			if len(agg) > 0:
				if key in [
					"episode_length",
					"episode_return",
					"mean_performance",
					"exploration_bonus",
					"e_mean",
					"e_std",
					"rnd_loss",
				]:
					agg_logs[key] = np.mean(agg)
				else:
					agg_logs[key] = np.array(agg)

		log_times.append(time.time())

		if config.debug:
			if len(log_times) == 1:
				print("Started logging")
			elif len(log_times) > 1:
				dt = log_times[-1] - log_times[-2]
				steps_between_updates = config.num_steps * config.num_envs * config.num_repeats
				sps = steps_between_updates / dt
				agg_logs["sps"] = sps

		agg_logs["update_step"] = update_step
		agg_logs["global_env_steps"] = int(global_env_steps)

		if config.use_wandb:
			wandb.log(agg_logs)


def advanced_create_log_dict(metric_info, config):
	"""Prepares the metric dictionary for logging. This version is designed to be
	more general and handle any key-value pairs passed from the training loop,
	including per-group metrics.

	Args:
	    metric_info: The dictionary of metrics produced by the training step.
	    config: The experiment configuration.

	Returns:
	    A dictionary ready to be logged.

	"""
	to_log = {}
	sum_achievements = 0

	# Iterate through all metrics passed from the training loop
	for k, v in metric_info.items():
		# Handle regular metrics and new per-group metrics (e.g., 'group_0/mean_return')
		if k == "returned_episode_returns":
			# k = "episode_return"
			continue
		elif k == "returned_episode_lengths":
			# k = "episode_length"
			continue
		elif k == "returned_episode" or k == "discount" or "achievements" in str(k).lower():
			continue
		to_log[k] = v
		# Special handling for achievements to calculate a total score
		if "skills" in str(k).lower():
			sum_achievements += v / 100.0

	to_log["achievements"] = sum_achievements

	return to_log


def advanced_batch_log(update_step, log, config):
	"""Aggregates logs from multiple independent runs (if config.num_repeats > 1)
	and sends the final averaged metrics to wandb.
	"""
	update_step = int(update_step)
	if update_step not in batch_logs:
		batch_logs[update_step] = []

	batch_logs[update_step].append(log)

	# Only proceed to log if we have collected results from all parallel runs
	if len(batch_logs[update_step]) == config.num_repeats:
		agg_logs = {}
		# Iterate through all keys in the first log as a template
		for key in batch_logs[update_step][0].keys():
			agg = []
			# Collect the value for this key from all parallel runs
			for i in range(config.num_repeats):
				val = batch_logs[update_step][i].get(key)
				if val is not None and not jnp.isnan(val):
					agg.append(val)

			if len(agg) > 0:
				# More robustly check if the item is a number that can be averaged.
				# This will correctly handle global stats AND our new per-group stats.
				if isinstance(agg[0], (int, float, np.number, jnp.number)):
					agg_logs[key] = np.mean(agg)
				else:
					# If it's not a simple number (e.g., an array), just keep the first one
					agg_logs[key] = np.array(agg[0])

		log_times.append(time.time())

		if config.debug:
			if len(log_times) > 1:
				dt = log_times[-1] - log_times[-2]
				# Calculate SPS based on total envs across all repeats
				steps_between_updates = config.num_steps * config.num_envs * config.num_repeats
				sps = steps_between_updates / dt
				agg_logs["sps"] = sps

		if config.use_wandb:
			wandb.log(agg_logs)
