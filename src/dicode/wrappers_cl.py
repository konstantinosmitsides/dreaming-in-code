from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


class GymnaxWrapper:
	"""Base class for Gymnax wrappers."""

	def __init__(self, env):
		self._env = env

	# provide proxy access to regular attributes of wrapped object
	def __getattr__(self, name):
		return getattr(self._env, name)


@struct.dataclass
class LogEnvState:
	env_state: Any
	episode_returns: jnp.ndarray
	episode_lengths: jnp.ndarray
	running_original_return: jnp.ndarray



class MultiTaskOptimisticLogWrapperAllTasks(GymnaxWrapper):
	"""A wrapper that handles optimistic resets and per-task logging for a
	vmapped, multi-task environment.
	"""

	def __init__(self, env, num_envs: int, num_tasks: int, reset_ratio: int, task_embeddings=None):
		super().__init__(env)
		self.num_envs = num_envs
		self.num_tasks = num_tasks
		self.reset_ratio = reset_ratio
		assert num_envs % reset_ratio == 0, "Reset ratio must perfectly divide num envs."
		self.num_resets = self.num_envs // reset_ratio

		# 1. Calculate how many times the task list needs to be repeated to cover all envs.
		num_repeats = (self.num_envs + self.num_tasks - 1) // self.num_tasks

		# 2. Create an array that is long enough by tiling.
		task_ids_long = jnp.tile(jnp.arange(self.num_tasks), num_repeats)

		# 3. Slice the array to the exact length required.
		self.task_ids = task_ids_long[: self.num_envs]
		self.task_embeddings = task_embeddings
		# This wrapper manages the vmapping internally
		self.vmapped_step = jax.vmap(self._env.step_env, in_axes=(0, 0, 0, None, None))
		self.vmapped_reset = jax.vmap(self._env.reset_env, in_axes=(0, None, 0, None))

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, key, params=None):
		# This reset is only called once at the beginning of training
		key, _rng = jax.random.split(key)
		reset_rngs = jax.random.split(_rng, self.num_envs)

		obs, env_state = self.vmapped_reset(reset_rngs, params, self.task_ids, self.task_embeddings)
		state = LogEnvState(
			env_state=env_state,
			episode_returns=jnp.zeros(self.num_envs),
			episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.int32),
			running_original_return=jnp.zeros(self.num_envs, dtype=jnp.int32),
		)
		return obs, state

	@partial(jax.jit, static_argnums=(0, 4))
	def step(self, key, state, action, params=None):
		key, step_rng, reset_rng = jax.random.split(key, 3)
		step_rngs = jax.random.split(step_rng, self.num_envs)

		obs, env_state, reward, done, info = self.vmapped_step(
			step_rngs, state.env_state, action, params, self.task_embeddings
		)

		# --- Logging Logic ---
		new_episode_return = state.episode_returns + reward
		new_episode_length = state.episode_lengths + 1
		info["task_id"] = state.env_state.task_id

		real_done = done & (new_episode_length > 1)

		info["returned_episode"] = real_done  # Use the filtered signal for logging
		info["returned_episode_returns"] = new_episode_return * real_done
		info["returned_episode_lengths"] = new_episode_length * real_done

		# info["returned_episode"] = done  # Use the filtered signal for logging
		# info["returned_episode_returns"] = new_episode_return * done
		# info["returned_episode_lengths"] = new_episode_length * done
		# --- NEW, CORRECT, AND EFFICIENT RESET LOGIC ---

		# 1. Find the indices of all environments that are done.
		#    The `size` argument ensures `done_indices` always has a fixed shape.
		#    We fill with an out-of-bounds index to safely ignore padded values.
		done_indices_padded = jnp.where(done, jnp.arange(self.num_envs), self.num_envs)
		done_indices = jnp.sort(done_indices_padded)  # Sort to bring valid indices to the front

		# 2. Select the first `self.num_resets` of the done indices.
		#    This is now a simple, static slice.
		indices_to_reset = done_indices[: self.num_resets]
		tasks_to_reset = self.task_ids[indices_to_reset]
		if self.task_embeddings is not None:
			task_embeddings_to_reset = self.task_embeddings
		else:
			task_embeddings_to_reset = None

		# 4. Perform a small, targeted reset for this FIXED-SIZE batch.
		reset_rngs = jax.random.split(reset_rng, self.num_resets)
		new_obs, new_states = self.vmapped_reset(
			reset_rngs, params, tasks_to_reset, task_embeddings_to_reset
		)

		# 5. Create a mask to apply the updates only where they are valid.
		#    An update is valid if the index corresponds to a real "done" environment.
		update_mask = indices_to_reset < self.num_envs

		# final_obs = jax.tree.map(
		# 	lambda old, new: old.at[indices_to_reset].set(
		# 		# Only use the new state if the update is valid,
		# 		# otherwise, keep the old value at that index.
		# 		jnp.where(
		# 			update_mask.reshape(-1, *([1] * (new.ndim - 1))), new, old[indices_to_reset]
		# 		)
		# 	),
		# 	obs,  # The original, full-sized state array
		# 	new_obs,  # The new, smaller array of reset states
		# )

		# # 6. Scatter the new states back into the main batch, using the mask.
		# final_env_state = jax.tree.map(
		# 	lambda old, new: old.at[indices_to_reset].set(
		# 		# Only use the new state if the update is valid,
		# 		# otherwise, keep the old value at that index.
		# 		jnp.where(
		# 			update_mask.reshape(-1, *([1] * (new.ndim - 1))), new, old[indices_to_reset]
		# 		)
		# 	),
		# 	env_state,  # The original, full-sized state array
		# 	new_states,  # The new, smaller array of reset states
		# )

		def apply_masked_update(old, new):
			# 1. Expand mask dims to match 'new' (e.g., (B,) -> (B, 1, 1))
			mask_broad = jnp.expand_dims(update_mask, axis=range(1, new.ndim))

			# 2. Apply the update
			return old.at[indices_to_reset].set(jnp.where(mask_broad, new, old[indices_to_reset]))

		# Run map once over a tuple of both states
		final_obs, final_env_state = jax.tree.map(
			apply_masked_update,
			(obs, env_state),  # Structure 1: The targets
			(new_obs, new_states),  # Structure 2: The sources
		)

		log_state = LogEnvState(
			env_state=final_env_state,
			episode_returns=new_episode_return * (1 - done),
			episode_lengths=new_episode_length * (1 - done),
			running_original_return=jnp.zeros(self.num_envs, dtype=jnp.int32),
		)

		return final_obs, log_state, reward, done, info


class DistributedMultiTaskOptimisticLogWrapper(GymnaxWrapper):
	"""A wrapper that handles optimistic resets and per-task logging for a
	vmapped, multi-task environment.
	"""

	def __init__(
		self,
		env,
		rng: jax.Array,
		num_envs: int,
		num_tasks: int,
		reset_ratio: int,
		task_distribution_proportions: jnp.ndarray,
		task_embeddings=None,
	):
		super().__init__(env)
		self.num_envs = num_envs
		self.num_tasks = num_tasks
		self.reset_ratio = reset_ratio
		assert num_envs % reset_ratio == 0, "Reset ratio must perfectly divide num envs."
		self.num_resets = self.num_envs // reset_ratio

		# 1. Validate the distribution
		assert len(task_distribution_proportions) == self.num_tasks, (
			f"Distribution length ({len(task_distribution_proportions)}) must match num_tasks ({self.num_tasks})"
		)
		assert jnp.isclose(jnp.sum(task_distribution_proportions), 1.0), (
			f"Proportions must sum to 1.0, but sum to {jnp.sum(task_distribution_proportions)}"
		)

		# 2. Calculate the integer number of envs for each task
		task_counts_float = task_distribution_proportions * self.num_envs
		task_counts_floor = jnp.floor(task_counts_float).astype(jnp.int32)

		# 3. Distribute the remainder (due to flooring) to the first 'remainder' tasks
		#    This ensures the total sum is exactly num_envs
		remainder = self.num_envs - jnp.sum(task_counts_floor)
		correction = (jnp.arange(self.num_tasks) < remainder).astype(jnp.int32)
		task_counts = task_counts_floor + correction

		# 4. Final check
		assert jnp.sum(task_counts) == self.num_envs, (
			"Calculated task counts do not sum to total num_envs"
		)

		# 5. Create the final task_ids array by repeating each task index
		#    e.g., [0, 0, ..., 1, 1, ..., 2, 2, ...]
		task_indices = jnp.arange(self.num_tasks)
		self.task_ids = jnp.repeat(task_indices, task_counts, total_repeat_length=self.num_envs)
		self.task_ids = jax.random.permutation(rng, self.task_ids)

		self.task_embeddings = task_embeddings
		# This wrapper manages the vmapping internally
		self.vmapped_step = jax.vmap(self._env.step_env, in_axes=(0, 0, 0, None, None))
		self.vmapped_reset = jax.vmap(self._env.reset_env, in_axes=(0, None, 0, None))

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, key, params=None):
		# This reset is only called once at the beginning of training
		key, _rng = jax.random.split(key)
		reset_rngs = jax.random.split(_rng, self.num_envs)

		obs, env_state = self.vmapped_reset(reset_rngs, params, self.task_ids, self.task_embeddings)
		state = LogEnvState(
			env_state=env_state,
			episode_returns=jnp.zeros(self.num_envs),
			episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.int32),
			running_original_return=jnp.zeros(self.num_envs, dtype=jnp.int32),
		)
		return obs, state

	@partial(jax.jit, static_argnums=(0, 4))
	def step(self, key, state, action, params=None):
		key, step_rng, reset_rng = jax.random.split(key, 3)
		step_rngs = jax.random.split(step_rng, self.num_envs)

		obs, env_state, reward, done, info = self.vmapped_step(
			step_rngs, state.env_state, action, params, self.task_embeddings
		)

		# --- Logging Logic ---
		new_episode_return = state.episode_returns + reward
		new_episode_length = state.episode_lengths + 1
		info["task_id"] = state.env_state.task_id

		real_done = done & (new_episode_length > 1)

		info["returned_episode"] = real_done  # Use the filtered signal for logging
		info["returned_episode_returns"] = new_episode_return * real_done
		info["returned_episode_lengths"] = new_episode_length * real_done

		# --- NEW, CORRECT, AND EFFICIENT RESET LOGIC ---

		# 1. Find the indices of all environments that are done.
		#    The `size` argument ensures `done_indices` always has a fixed shape.
		#    We fill with an out-of-bounds index to safely ignore padded values.
		done_indices_padded = jnp.where(done, jnp.arange(self.num_envs), self.num_envs)
		done_indices = jnp.sort(done_indices_padded)  # Sort to bring valid indices to the front

		# 2. Select the first `self.num_resets` of the done indices.
		#    This is now a simple, static slice.
		indices_to_reset = done_indices[: self.num_resets]
		tasks_to_reset = self.task_ids[indices_to_reset]
		if self.task_embeddings is not None:
			task_embeddings_to_reset = self.task_embeddings
		else:
			task_embeddings_to_reset = None

		# 4. Perform a small, targeted reset for this FIXED-SIZE batch.
		reset_rngs = jax.random.split(reset_rng, self.num_resets)
		new_obs, new_states = self.vmapped_reset(
			reset_rngs, params, tasks_to_reset, task_embeddings_to_reset
		)

		# 5. Create a mask to apply the updates only where they are valid.
		#    An update is valid if the index corresponds to a real "done" environment.
		update_mask = indices_to_reset < self.num_envs

		# 6. Scatter the new states back into the main batch, using the mask.
		# final_env_state = jax.tree.map(
		# 	lambda old, new: old.at[indices_to_reset].set(
		# 		# Only use the new state if the update is valid,
		# 		# otherwise, keep the old value at that index.
		# 		jnp.where(
		# 			update_mask.reshape(-1, *([1] * (new.ndim - 1))), new, old[indices_to_reset]
		# 		)
		# 	),
		# 	env_state,  # The original, full-sized state array
		# 	new_states,  # The new, smaller array of reset states
		# )

		def apply_masked_update(old, new):
			# 1. Expand mask dims to match 'new' (e.g., (B,) -> (B, 1, 1))
			mask_broad = jnp.expand_dims(update_mask, axis=range(1, new.ndim))

			# 2. Apply the update
			return old.at[indices_to_reset].set(jnp.where(mask_broad, new, old[indices_to_reset]))

		# Run map once over a tuple of both states
		final_obs, final_env_state = jax.tree.map(
			apply_masked_update,
			(obs, env_state),  # Structure 1: The targets
			(new_obs, new_states),  # Structure 2: The sources
		)

		log_state = LogEnvState(
			env_state=final_env_state,
			episode_returns=new_episode_return * (1 - done),
			episode_lengths=new_episode_length * (1 - done),
			running_original_return=jnp.zeros(self.num_envs, dtype=jnp.int32),

		)

		return final_obs, log_state, reward, done, info
