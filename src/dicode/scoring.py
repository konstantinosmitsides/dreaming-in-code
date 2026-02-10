# FILE: dicode/scoring.py

import jax
import numpy as np
from craftax.craftax.constants import Achievement
from omegaconf import DictConfig

# --- Helper function to get achievement names (Unchanged) ---


def get_achievement_names():
	"""Returns a list of all achievement names in lowercase."""
	return [ach.name.lower() for ach in Achievement]


# --- Main "Smart Calculator" Framework (MODIFIED) ---


def calculate_scores_from_snapshot(
	scoring_window_data: dict,
	num_tasks: int,
	task_achievement_mask: np.ndarray,
	task_completed_mask: np.ndarray,
	config: DictConfig,
	force_include_achievements_indices: list[int] = None,
) -> dict:
	"""The main "Smart Calculator" function.

	Orchestrates the calculation of all metrics:
	1. Calculates base metrics (SR, AchSRs) for all tasks.
	2. Calculates the specific `priority_score` for all tasks based on config.
	3. Merges them into a final metrics dictionary.
	"""
	# 1. Convert all JAX arrays in the Pytree to NumPy arrays *once*.
	# This is the most efficient way to move data from device to host.
	scoring_window_data_np = jax.device_get(scoring_window_data)

	# Pull from scoring_window_data_np
	traj_batch = scoring_window_data_np.get("traj_batch")

	# Pull from scoring_window_data_np (no np.asarray needed)
	advantages = scoring_window_data_np.get("advantages")

	# Extract info from traj_batch
	info = traj_batch.info
	task_ids = info["task_id"]
	returned_episode = info["returned_episode"]
	is_success = info["is_success"]
	rewards = traj_batch.reward
	values = traj_batch.value
	episode_lengths = info["returned_episode_lengths"]
	episode_returns = info["returned_episode_returns"]

	# 2. Always calculate base metrics for logging and categorization
	print(f"  - [Scoring] Calculating base metrics (SR, AchSR) for {num_tasks} tasks...")
	base_metrics = _calculate_base_metrics(
		info,
		task_ids,
		returned_episode,
		episode_returns,
		is_success,
		num_tasks,
		task_achievement_mask,
		task_completed_mask,
		config,
		force_include_achievements_indices,
	)

	print(
		"--------------------------------------------------------------------------------------------------"
	)
	print(f"DEBUG: base metrics:{base_metrics}")
	print(
		"--------------------------------------------------------------------------------------------------"
	)
	# 3. Calculate the priority score based on the config
	print(f"  - [Scoring] Calculating priority scores using '{config.dicode_manager.score_function}'...")
	priority_scores = _calculate_priority_scores(
		config.dicode_manager.score_function,
		base_metrics,
		task_ids,
		returned_episode,
		advantages,
		rewards,
		values,
		num_tasks,
	)

	# 4. Merge base metrics and priority scores
	final_metrics = {}
	for task_idx_str, metrics in base_metrics.items():
		task_idx = int(task_idx_str)
		metrics["priority_score"] = priority_scores.get(task_idx, 0.0)
		final_metrics[task_idx_str] = metrics

	print(
		"--------------------------------------------------------------------------------------------------"
	)
	print(f"DEBUG: final metrics:{final_metrics}")
	print(
		"--------------------------------------------------------------------------------------------------"
	)

	return final_metrics


def _calculate_base_metrics(
	info: dict,
	task_ids: np.ndarray,
	returned_episode: np.ndarray,
	episode_returns: np.ndarray,
	is_success: np.ndarray,
	num_tasks: int,
	task_achievement_mask: np.ndarray,
	task_completed_mask: np.ndarray,
	config: DictConfig,
	force_include_achievements_indices: list[int] = None,
) -> dict:
	"""Calculates Success Rate (SR), Achievement SRs, and Average Episode Length
	from the raw snapshot data.
	"""
	# 1. Stack all achievement arrays
	ach_names = get_achievement_names()
	all_achievements_arrays = np.stack(
		[info[f"Achievements/{name}"] for name in ach_names], axis=-1
	)  # Shape: (T, B, num_achievements)

	episode_lengths = info["returned_episode_lengths"]

	final_metrics = {}
	force_indices = (
		set(force_include_achievements_indices) if force_include_achievements_indices else set()
	)

	# 2. Loop over each task and parse the *entire* snapshot
	for task_idx in range(num_tasks):
		# Create masks for this specific task
		task_mask = task_ids == task_idx
		task_done_mask = task_mask & returned_episode

		num_finished = np.sum(task_done_mask)
		mean_return = np.sum(episode_returns * task_done_mask) / num_finished
		mean_return_percentage = mean_return / 226.0 * 100.0
		num_successes = np.sum(is_success & task_done_mask)

		# --- Calculate SR ---
		if num_finished > 0:
			sr = num_successes / num_finished
		else:
			sr = -1.0  # Use -1.0 to indicate no episodes were finished

		# --- Calculate Average Episode Length ---
		if num_finished > 0:
			total_length = np.sum(episode_lengths * task_done_mask)
			avg_len = total_length / num_finished
		else:
			avg_len = 0.0

		# --- Calculate Achievement SRs ---
		safe_num_finished = np.maximum(1.0, num_finished)
		broadcast_mask = task_done_mask[..., None]
		summed_achievements = np.sum(all_achievements_arrays * broadcast_mask, axis=(0, 1))

		# Get the mask *for this specific task*
		relevant_ach_mask_for_task = task_achievement_mask[task_idx]
		completed_ach_mask_for_task = task_completed_mask[task_idx]
		ach_srs_percentages = (
			summed_achievements / safe_num_finished
		)  # / 100.0 removed that, save percentages instead

		ach_sr_dict = {}
		if config.dicode_manager.mode == "reward":
			for i, name in enumerate(ach_names):
				# Only include this achievement if it's in the
				# "relevant_achievements" list for this task.
				if relevant_ach_mask_for_task[i]:
					ach_sr_val = ach_srs_percentages[i]
					ach_sr_dict[name] = float(ach_sr_val if sr >= 0 else -1.0)

				if completed_ach_mask_for_task[i]:
					continue

		else:
			if sr >= 0:
				for i, name in enumerate(ach_names):
					# The "if relevant_ach_mask_for_task[i]:" check is REMOVED.
					# We now report the SR for *all* achievements.

					if completed_ach_mask_for_task[i]:
						continue

					ach_sr_val = ach_srs_percentages[i]

					# We only report skills with SR > 0
					# to avoid cluttering logs with hundreds of zero-value skills.
					# Unless forced (e.g. for original task)
					if ach_sr_val > 0 or task_idx in force_indices:
						ach_sr_dict[name] = float(ach_sr_val if sr >= 0 else -1.0)

		# 3. Store all metrics for this task
		final_metrics[str(task_idx)] = {
			"sr": float(sr),
			"achievement_srs": ach_sr_dict,
			"average_episode_length": float(avg_len),
			"mean_return_percentage": float(mean_return_percentage),
			"mean_return": float(mean_return),
			"lp": np.nan,  # LP is not calculated
		}

	return final_metrics


# --- NEW: Priority Score Dispatcher ---


def _calculate_priority_scores(
	score_function: str,
	base_metrics: dict,
	task_ids: np.ndarray,
	dones: np.ndarray,
	advantages: np.ndarray,
	rewards: np.ndarray,
	values: np.ndarray,
	num_tasks: int,
) -> dict:
	"""Dispatcher that calculates the `priority_score` for all tasks."""
	if score_function == "learnability":
		# Learnability is just sr * (1-sr)
		priority_scores = {}
		for task_idx in range(num_tasks):
			sr = base_metrics[str(task_idx)]["sr"]
			# if sr == 0.0:
			# 	clipped_sr = 0.1
			# else:
			clipped_sr = np.clip(sr, 0.0, 1.0) if sr >= 0 else 0.0
			priority_scores[task_idx] = clipped_sr * (1.0 - clipped_sr)
		return priority_scores

	elif score_function == "pvl":
		return _compute_pvl_scores_from_snapshot(dones, advantages, task_ids, num_tasks)

	elif score_function == "max_mc":
		return _compute_max_mc_scores_from_snapshot(dones, rewards, values, task_ids, num_tasks)

	else:
		raise ValueError(f"Unknown score_function: {score_function}")


# --- NEW: PVL and MaxMC Implementations ---


def _compute_pvl_scores_from_snapshot(
	dones: np.ndarray, advantages: np.ndarray, task_ids: np.ndarray, num_tasks: int
) -> dict:
	"""NumPy-based implementation of Positive Value Loss (PVL).
	This is a Python port of the logic in `positive_value_loss`.
	"""
	T, B = dones.shape

	# Per-environment accumulators
	ep_pos_adv_sum = np.zeros(B, dtype=np.float32)
	ep_steps = np.zeros(B, dtype=np.int32)

	# Storage for all *episode* scores (not step scores)
	task_episode_scores = [[] for _ in range(num_tasks)]

	# Iterate through time, then environments
	for t in range(T):
		for b in range(B):
			done = dones[t, b]
			adv = advantages[t, b]
			task_id = task_ids[t, b]

			ep_pos_adv_sum[b] += max(adv, 0.0)
			ep_steps[b] += 1

			if done:
				if ep_steps[b] > 0:
					# This is the "time_average" logic
					ep_score = ep_pos_adv_sum[b] / ep_steps[b]
					task_episode_scores[task_id].append(ep_score)

				# Reset accumulators for this env
				ep_pos_adv_sum[b] = 0.0
				ep_steps[b] = 0.0

	# Average the scores for each task
	final_priority_scores = {}
	for task_idx in range(num_tasks):
		scores = task_episode_scores[task_idx]
		if len(scores) > 0:
			final_priority_scores[task_idx] = np.mean(scores)
		else:
			# No episodes finished, return -inf as in the reference
			final_priority_scores[task_idx] = -np.inf

	return final_priority_scores


def _compute_max_mc_scores_from_snapshot(
	dones: np.ndarray, rewards: np.ndarray, values: np.ndarray, task_ids: np.ndarray, num_tasks: int
) -> dict:
	"""NumPy-based implementation of "Batch" Max Monte-Carlo (MaxMC).
	This is a port of the `max_mc` logic.

	Note: This computes MaxMC based *only* on the max returns
	seen *within this snapshot*, not historical max returns.
	"""
	T, B = dones.shape

	# --- Pass 1: Find the max return for each task *in this batch* ---
	ep_returns = np.zeros(B, dtype=np.float32)
	# Use -inf default so any return is larger
	task_max_returns = np.full(num_tasks, -np.inf, dtype=np.float32)

	for t in range(T):
		for b in range(B):
			done = dones[t, b]
			reward = rewards[t, b]
			task_id = task_ids[t, b]

			ep_returns[b] += reward

			if done:
				task_max_returns[task_id] = np.maximum(task_max_returns[task_id], ep_returns[b])
				ep_returns[b] = 0.0  # Reset

	# --- Pass 2: Calculate the time-averaged "regret" (MaxReturn - Value) ---
	ep_regret_sum = np.zeros(B, dtype=np.float32)
	ep_steps = np.zeros(B, dtype=np.int32)
	task_episode_scores = [[] for _ in range(num_tasks)]

	for t in range(T):
		for b in range(B):
			done = dones[t, b]
			value = values[t, b]
			task_id = task_ids[t, b]

			# Get the max return for this task (found in Pass 1)
			task_max_ret = task_max_returns[task_id]

			# Only accumulate if the max return is valid (not -inf)
			if task_max_ret > -np.inf:
				ep_regret_sum[b] += task_max_ret - value

			ep_steps[b] += 1

			if done:
				if ep_steps[b] > 0 and task_max_ret > -np.inf:
					ep_score = ep_regret_sum[b] / ep_steps[b]
					task_episode_scores[task_id].append(ep_score)

				# Reset accumulators
				ep_regret_sum[b] = 0.0
				ep_steps[b] = 0

	# Average the scores for each task
	final_priority_scores = {}
	for task_idx in range(num_tasks):
		scores = task_episode_scores[task_idx]
		if len(scores) > 0:
			final_priority_scores[task_idx] = np.mean(scores)
		else:
			final_priority_scores[task_idx] = -np.inf

	return final_priority_scores
