from functools import partial
from typing import Any

import chex
import jax
import jax.lax as lax
import jax.numpy as jnp
from craftax.craftax.constants import OBS_DIM, Action, BlockType, ItemType
from craftax.craftax.craftax_state import EnvParams, EnvState, StaticEnvParams
from craftax.craftax.envs.common import log_achievements_to_info
from craftax.craftax.game_logic import craftax_step, is_game_over
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world
from flax import struct
from gymnasium import spaces
from jax.tree_util import tree_map

# from minicraftax_old.advanced_wrapper.config import EnvConfig


def get_flat_map_obs_shape():
	map_obs_shape = get_map_obs_shape()
	return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_map_obs_shape():
	num_mob_classes = 5
	num_mob_types = 8
	num_blocks = len(BlockType)
	num_items = len(ItemType)

	return (
		OBS_DIM[0],
		OBS_DIM[1],
		num_blocks + num_items + num_mob_classes * num_mob_types + 1,
	)


def get_inventory_obs_shape():
	return 51


class GymnaxWrapper:
	"""Base class for Gymnax wrappers."""

	def __init__(self, env):
		self._env = env

	# provide proxy access to regular attributes of wrapped object
	def __getattr__(self, name):
		return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
	"""Batches reset and step functions"""

	def __init__(self, env, num_envs: int):
		super().__init__(env)

		self.num_envs = num_envs

		self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
		self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, rng, params=None):
		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_envs)
		obs, env_state = self.reset_fn(rngs, params)
		return obs, env_state

	@partial(jax.jit, static_argnums=(0, 4))
	def step(self, rng, state, action, params=None):
		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_envs)
		obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

		return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
	"""Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

	def __init__(self, env):
		super().__init__(env)

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, key, params=None):
		return self._env.reset(key, params)

	@partial(jax.jit, static_argnums=(0, 4))
	def step(self, rng, state, action, params=None):
		rng, _rng = jax.random.split(rng)
		obs_st, state_st, reward, done, info = self._env.step(_rng, state, action, params)

		rng, _rng = jax.random.split(rng)
		obs_re, state_re = self._env.reset(_rng, params)

		# Auto-reset environment based on termination
		def auto_reset(done, state_re, state_st, obs_re, obs_st):
			state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
			obs = jax.lax.select(done, obs_re, obs_st)

			return obs, state

		obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

		return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper_(GymnaxWrapper):
	"""Provides efficient 'optimistic' resets.
	The wrapper also necessarily handles the batching of environment steps and resetting.
	reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
	chance of duplicate resets.
	"""

	def __init__(self, env, num_envs: int, reset_ratio: int):
		super().__init__(env)

		self.num_envs = num_envs
		self.reset_ratio = reset_ratio
		assert num_envs % reset_ratio == 0, "Reset ratio must perfectly divide num envs."
		self.num_resets = self.num_envs // reset_ratio

		self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
		self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, rng, params=None):
		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_envs)
		obs, env_state = self.reset_fn(rngs, params)
		return obs, env_state

	@partial(jax.jit, static_argnums=(0, 4))
	def step(self, rng, state, action, params=None):
		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_envs)
		obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_resets)
		obs_re, state_re = self.reset_fn(rngs, params)

		rng, _rng = jax.random.split(rng)
		reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

		being_reset = jax.random.choice(
			_rng,
			jnp.arange(self.num_envs),
			shape=(self.num_resets,),
			p=done,
			replace=False,
		)
		reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

		obs_re = obs_re[reset_indexes]
		state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

		# Auto-reset environment based on termination
		def auto_reset(done, state_re, state_st, obs_re, obs_st):
			state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
			obs = jax.lax.select(done, obs_re, obs_st)

			return state, obs

		state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

		return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
	"""Provides efficient 'optimistic' resets.
	The wrapper also necessarily handles the batching of environment steps and resetting.
	reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
	chance of duplicate resets.
	"""

	def __init__(self, env, num_envs: int, reset_ratio: int):
		super().__init__(env)

		self.num_envs = num_envs
		self.reset_ratio = reset_ratio
		assert num_envs % reset_ratio == 0, "Reset ratio must perfectly divide num envs."
		self.num_resets = self.num_envs // reset_ratio

		self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
		self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, rng, params=None):
		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_envs)
		obs, env_state = self.reset_fn(rngs, params)
		return obs, env_state

	@partial(jax.jit, static_argnums=(0, 4))
	def step(self, rng, state, action, params=None):
		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_envs)
		obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_resets)
		obs_re, state_re = self.reset_fn(rngs, params)

		rng, _rng = jax.random.split(rng)
		reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

		being_reset = jax.random.choice(
			_rng,
			jnp.arange(self.num_envs),
			shape=(self.num_resets,),
			p=done,
			replace=False,
		)
		reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

		obs_re = obs_re[reset_indexes]
		state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

		# Auto-reset environment based on termination
		def auto_reset(done, state_re, state_st, obs_re, obs_st):
			state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
			obs = jax.lax.select(done, obs_re, obs_st)

			return state, obs

		state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

		return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
	env_state: Any
	episode_returns: float
	episode_lengths: int
	returned_episode_returns: float
	returned_episode_lengths: int
	timestep: int


class LogWrapper(GymnaxWrapper):
	"""Log the episode returns and lengths."""

	def __init__(self, env):
		super().__init__(env)

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, key: chex.PRNGKey, params=None):
		obs, env_state = self._env.reset(key, params)
		state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
		return obs, state

	@partial(jax.jit, static_argnums=(0, 4))
	def step(
		self,
		key: chex.PRNGKey,
		state,
		action: int | float,
		params=None,
	):
		obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
		new_episode_return = state.episode_returns + reward
		new_episode_length = state.episode_lengths + 1
		state = LogEnvState(
			env_state=env_state,
			episode_returns=new_episode_return * (1 - done),
			episode_lengths=new_episode_length * (1 - done),
			returned_episode_returns=state.returned_episode_returns * (1 - done)
			+ new_episode_return * done,
			returned_episode_lengths=state.returned_episode_lengths * (1 - done)
			+ new_episode_length * done,
			timestep=state.timestep + 1,
		)
		info["returned_episode_returns"] = state.returned_episode_returns
		info["returned_episode_lengths"] = state.returned_episode_lengths
		info["timestep"] = state.timestep
		info["returned_episode"] = done
		return obs, state, reward, done, info


class EnvironmentNoAutoReset:
	"""Similar to the base Gymnax environment but without auto-resets."""

	@property
	def default_params(self):
		return NotImplementedError

	@property
	def default_config(self):
		return NotImplementedError

	@partial(jax.jit, static_argnums=(0, 4))
	def step(
		self,
		key: chex.PRNGKey,
		state,
		action: int | float,
		params=None,
		config=None,
	):
		"""Performs step transitions in the environment."""
		# Use default env parameters if no others specified
		if params is None:
			params = self.default_params
		if config is None:
			config = self.default_config
		obs, state, reward, done, info = self.step_env(key, state, action, params, config)
		return obs, state, reward, done, info

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, key: chex.PRNGKey, params=None, config=None):
		"""Performs resetting of environment."""
		# Use default env parameters if no others specified
		if params is None:
			params = self.default_params
		if config is None:
			config = self.default_config
		obs, state = self.reset_env(key, params, config)
		return obs, state

	def step_env(
		self,
		key: chex.PRNGKey,
		state,
		action: int | float,
		params,
		config,
	):
		"""Environment-specific step transition."""
		raise NotImplementedError

	def reset_env(self, key: chex.PRNGKey, params, config):
		"""Environment-specific reset."""
		raise NotImplementedError

	def get_obs(self, state) -> chex.Array:
		"""Applies observation function to state."""
		raise NotImplementedError

	def is_terminal(self, state, params) -> bool:
		"""Check whether state transition is terminal."""
		raise NotImplementedError

	def discount(self, state, params) -> float:
		"""Return a discount of zero if the episode has terminated."""
		return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

	@property
	def name(self) -> str:
		"""Environment name."""
		return type(self).__name__

	@property
	def num_actions(self) -> int:
		"""Number of actions possible in environment."""
		raise NotImplementedError

	def action_space(self, params):
		"""Action space of the environment."""
		raise NotImplementedError

	def observation_space(self, params):
		"""Observation space of the environment."""
		raise NotImplementedError

	def state_space(self, params):
		"""State space of the environment."""
		raise NotImplementedError


# class CraftaxSymbolicEnvNoAutoReset(EnvironmentNoAutoReset):
# 	def __init__(self, static_env_params: StaticEnvParams | None = None):
# 		super().__init__()

# 		if static_env_params is None:
# 			static_env_params = self.default_static_params()
# 		self.static_env_params = static_env_params

# 	@property
# 	def default_params(self) -> EnvParams:
# 		return EnvParams()

# 	@property
# 	def default_config(self) -> EnvConfig:
# 		return EnvConfig()

# 	@staticmethod
# 	def default_static_params() -> StaticEnvParams:
# 		return StaticEnvParams()

# 	def step_env(
# 		self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams, config: EnvConfig
# 	) -> tuple[chex.Array, EnvState, float, bool, dict]:
# 		state, reward = craftax_step(rng, state, action, params, self.static_env_params, config)

# 		done = self.is_terminal(state, params, config)
# 		info = log_achievements_to_info(state, done)
# 		info["discount"] = self.discount(state, params, config)

# 		return (
# 			lax.stop_gradient(self.get_obs(state)),
# 			lax.stop_gradient(state),
# 			reward,
# 			done,
# 			info,
# 		)

# 	def reset_env(
# 		self, rng: chex.PRNGKey, params: EnvParams, config: EnvConfig
# 	) -> tuple[chex.Array, EnvState]:
# 		rng, _rng = jax.random.split(rng)
# 		state = generate_world(_rng, params, self.static_env_params, config)

# 		return self.get_obs(state), state

# 	def get_obs(self, state: EnvState) -> chex.Array:
# 		pixels = render_craftax_symbolic(state)
# 		return pixels

# 	def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
# 		return is_game_over(state, params, self.static_env_params)

# 	@property
# 	def name(self) -> str:
# 		return "Craftax-Symbolic-NoAutoReset-v1"

# 	@property
# 	def num_actions(self) -> int:
# 		return len(Action)

# 	def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
# 		return spaces.Discrete(len(Action))

# 	def observation_space(self, params: EnvParams) -> spaces.Box:
# 		flat_map_obs_shape = get_flat_map_obs_shape()
# 		inventory_obs_shape = get_inventory_obs_shape()

# 		obs_shape = flat_map_obs_shape + inventory_obs_shape

# 		return spaces.Box(
# 			0.0,
# 			1.0,
# 			(obs_shape,),
# 			dtype=jnp.float32,
# 		)


class AdvancedOptimisticResetVecEnvWrapper(GymnaxWrapper):
	"""Provides efficient 'optimistic' resets. A fixed number of environments are
	prepared for reset, but only those that are 'done' are actually replaced.
	"""

	def __init__(self, env, num_envs: int, reset_ratio: int):
		super().__init__(env)

		self.num_envs = num_envs
		self.reset_ratio = reset_ratio
		assert num_envs % reset_ratio == 0, "Reset ratio must perfectly divide num envs."
		self.num_resets = self.num_envs // self.reset_ratio

		self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None, 0))
		self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None, 0))

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, rng, params=None, config=None):
		"""Resets all environments in the batch."""
		rng, _rng = jax.random.split(rng)
		rngs = jax.random.split(_rng, self.num_envs)
		obs, env_state = self.reset_fn(rngs, params, config)
		return obs, env_state

	@partial(jax.jit, static_argnums=(0, 4))
	def step(self, rng, state, action, params=None, config=None):
		"""Steps all environments and resets a subset based on the 'done' flag."""
		# 1. Step all environments as usual.
		rng, step_rng = jax.random.split(rng)
		step_rngs = jax.random.split(step_rng, self.num_envs)
		obs_st, state_st, reward, done, info = self.step_fn(
			step_rngs, state, action, params, config
		)

		# --- "Optimistic" Reset Logic with Fix ---

		# 2. Select which environments to *prepare* for reset.
		# This prioritizes 'done' environments but will choose running ones if needed.
		rng, choice_rng = jax.random.split(rng)
		scores = done.astype(jnp.float32) + jax.random.uniform(choice_rng, shape=(self.num_envs,))
		_, indices_to_prepare = jax.lax.top_k(scores, k=self.num_resets)

		# 3. Gather the specific configurations for the environments we are preparing to reset.
		# This is the main fix to prevent the size mismatch error.
		reset_configs = tree_map(lambda x: x[indices_to_prepare], config)

		# 4. Perform the resets for the selected environments.
		rng, reset_rng = jax.random.split(rng)
		reset_rngs = jax.random.split(reset_rng, self.num_resets)
		obs_re_compact, state_re_compact = self.reset_fn(reset_rngs, params, reset_configs)

		# 5. Create the full-size reset state/observation batches.
		# We scatter the compact reset states into a full batch, which will be used
		# by the final auto_reset function.
		def scatter_leaf(full_leaf, compact_leaf):
			return jnp.zeros_like(full_leaf).at[indices_to_prepare].set(compact_leaf)

		state_re = tree_map(scatter_leaf, state_st, state_re_compact)
		obs_re = scatter_leaf(obs_st, obs_re_compact)

		# 6. Final Auto-Reset Check (as in the original code).
		# This function checks the 'done' flag for each environment and only
		# replaces the state if the flag is True.
		def auto_reset(done, state_re, state_st, obs_re, obs_st):
			state = tree_map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
			obs = jax.lax.select(done, obs_re, obs_st)
			return state, obs

		# vmap the final check over all environments.
		state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

		return obs, state, reward, done, info


class AdvancedLogWrapper(GymnaxWrapper):
	"""A vectorized wrapper that logs episode returns and lengths for a batch of environments.
	This wrapper should enclose a batched environment.
	"""

	def __init__(self, env, **kwargs):
		super().__init__(env)
		self.num_envs = self._env.num_envs

	@partial(jax.jit, static_argnums=(0, 2))
	def reset(self, key: jax.Array, params=None, config=None):
		"""Resets the environment and initializes the logging state."""
		obs, env_state = self._env.reset(key, params, config)
		log_state = LogEnvState(
			env_state=env_state,
			episode_returns=jnp.zeros(self.num_envs),
			episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.int32),
			returned_episode_returns=jnp.zeros(self.num_envs),
			returned_episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.int32),
			timestep=jnp.zeros(self.num_envs, dtype=jnp.int32),
		)
		return obs, log_state

	@partial(jax.jit, static_argnums=(0, 4))
	def step(self, key: jax.Array, state: LogEnvState, action, params=None, config=None):
		"""Steps the environment and adds per-environment data to the info dict."""
		obs, env_state, reward, done, info = self._env.step(
			key, state.env_state, action, params, config
		)

		new_episode_returns = state.episode_returns + reward
		new_episode_lengths = state.episode_lengths + 1

		# Update the running totals
		log_state = state.replace(
			env_state=env_state,
			episode_returns=new_episode_returns * (1 - done),
			episode_lengths=new_episode_lengths * (1 - done),
			returned_episode_returns=jnp.where(
				done, new_episode_returns, state.returned_episode_returns
			),
			returned_episode_lengths=jnp.where(
				done, new_episode_lengths, state.returned_episode_lengths
			),
			timestep=state.timestep + 1,
		)

		# Add the keys the training loop expects to the info dict
		info["returned_episode"] = done
		info["returned_episode_returns"] = new_episode_returns * done
		info["returned_episode_lengths"] = new_episode_lengths * done

		return obs, log_state, reward, done, info


class EvaluationWrapper(GymnaxWrapper):
	"""A wrapper that concatenates a fixed embedding to every observation."""

	def __init__(self, env, embedding_to_concat):
		super().__init__(env)
		self.embedding = embedding_to_concat
		self.task_vector_size = embedding_to_concat.shape[-1]

	# Override the reset method
	def reset(self, key, params=None):
		# Get the original observation from the wrapped env
		obs, state = self._env.reset(key, params)
		# Concatenate the embedding before returning
		conditioned_obs = jnp.concatenate([obs, self.embedding], axis=-1)
		return conditioned_obs, state

	# Override the step method
	def step(self, key, state, action, params=None):
		# Get the original observation from the wrapped env
		obs, state, reward, done, info = self._env.step(key, state, action, params)
		# Concatenate the embedding before returning
		conditioned_obs = jnp.concatenate([obs, self.embedding], axis=-1)
		return conditioned_obs, state, reward, done, info

	# def observation_space(self, params: EnvParams) -> spaces.Box:
	#     """
	#     Returns the observation space, accounting for the concatenated task vector.
	#     """
	#     parent_space = super().observation_space(params)
	#     # parent_space = self._env.observation_space(params)

	#     new_obs_shape = (parent_space.shape[0] + self.task_vector_size,)

	#     # Handle scalar bounds for the parent's observation space
	#     if jnp.isscalar(parent_space.low):
	#         low_bound = jnp.full(parent_space.shape, parent_space.low, dtype=parent_space.dtype)
	#     else:
	#         low_bound = parent_space.low

	#     if jnp.isscalar(parent_space.high):
	#         high_bound = jnp.full(parent_space.shape, parent_space.high, dtype=parent_space.dtype)
	#     else:
	#         high_bound = parent_space.high

	#     task_low = jnp.full(self.task_vector_size, -jnp.inf, dtype=parent_space.dtype)
	#     task_high = jnp.full(self.task_vector_size, jnp.inf, dtype=parent_space.dtype)

	#     new_low = jnp.concatenate([low_bound, task_low])
	#     new_high = jnp.concatenate([high_bound, task_high])

	#     return spaces.Box(
	#         low=new_low,
	#         high=new_high,
	#         shape=new_obs_shape,
	#         dtype=parent_space.dtype
	#     )
