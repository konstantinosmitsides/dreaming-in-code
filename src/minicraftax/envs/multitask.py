# In src/minicraftax/envs/multitask.py
import jax
import jax.numpy as jnp
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement, Action

# Import from original library
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.envs.common import log_achievements_to_info
from craftax.craftax.game_logic import (
	boss_logic,
	calculate_inventory_achievements,
	cast_spell,
	clip_inventory_and_intrinsics,
	do_action,
	do_crafting,
	drink_potion,
	enchant,
	level_up_attributes,
	move_player,
	place_block,
	read_book,
	shoot_projectile,
)
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.util.game_logic_utils import calculate_light_level
from gymnax.environments import spaces
from jax import lax

# Import TaskParams from its new location
from minicraftax.craftax_state import EnvState

# Import NEW components
from minicraftax.envs.base import MiniCraftaxTrain, MiniCraftaxTrainR  # Base env we inherit from
from minicraftax.game_mechanics import (
	change_floor,
	spawn_mobs,
	update_mobs,
	update_plants,
	update_player_intrinsics,
)
from minicraftax.tasks.base_task import BaseTask, clamp_task_params  # Import clamping function


class MultiTaskMiniCraftaxEnv(MiniCraftaxTrain):
	"""A JAX-compatible environment for running multiple MiniCraftax tasks simultaneously.

	Allows for conditioning on task IDs, computing bonuses for task completion, and
	managing multiple task definitions within a single vectorized environment.

	Attributes:
		completion_bonus_scale (float): Scale factor for task completion bonus.
		completion_bonus_min (float): Minimum bonus for task completion.
		tasks (list): List of instantiated BaseTask objects.
		num_tasks (int): Total number of tasks.
	"""

	def __init__(
		self,
		task_classes: list[type[BaseTask]],
		static_env_params: StaticEnvParams,
		params: EnvParams,
		condition_on_task: bool = False,
		conditioning_type: str = "one_hot",
		embedding_size: int = 67,
		completion_bonus_scale: float = 2.0,
		completion_bonus_min: float = 20.0,
		bonus_type: str = "static",
		dynamic_bonus_k: float = 2.0,
	):
		self.completion_bonus_scale = completion_bonus_scale
		self.completion_bonus_min = completion_bonus_min
		self.bonus_type = bonus_type
		self.dynamic_bonus_k = dynamic_bonus_k

		self.tasks = [task_cls(static_env_params, params) for task_cls in task_classes]
		self.num_tasks = len(self.tasks)

		# Store world gen functions (JAX-switchable)
		self.world_gen_fns = tuple(t.generate_world for t in self.tasks)
		self.terminal_fns = tuple(t.is_terminal for t in self.tasks)
		self.success_fns = tuple(t.is_success for t in self.tasks)

		# --- Aggregate, CLAMP, and stack TaskParams ---
		self.all_task_params_raw = [t.get_task_params() for t in self.tasks]
		# Apply clamping during initialization
		self.all_task_params_clamped = [clamp_task_params(p) for p in self.all_task_params_raw]
		self.stacked_task_params = jax.tree.map(
			lambda *x: jnp.stack(x),
			*self.all_task_params_clamped,  # Stack the clamped parameters
		)
		# --- End TaskParams handling ---

		# --- NEW: PRE-CALCULATE SCALED BONUSES ---
		achievement_coeff = ACHIEVEMENT_REWARD_MAP
		task_intrinsic_rewards = []
		for task in self.tasks:
			if task.relevant_achievements:
				relevant_indices = jnp.array([ach.value for ach in task.relevant_achievements])
				mask = jnp.zeros_like(achievement_coeff).at[relevant_indices].set(1.0)
				task_reward = (mask * achievement_coeff).sum()
				task_intrinsic_rewards.append(task_reward)
			else:
				task_intrinsic_rewards.append(0.0)

		# Store this as a JAX array for fast lookup in step_env
		self.task_intrinsic_rewards = jnp.array(task_intrinsic_rewards)

		self.condition_on_task = condition_on_task
		self.conditioning_type = conditioning_type
		# self.task_vector_size = embedding_size
		if condition_on_task:
			# if conditioning_type == "one_hot":
			# 	self.task_vector_size = self.num_tasks
			if conditioning_type in ["embedding", "one_hot"]:
				self.task_vector_size = embedding_size
			else:
				raise ValueError(f"Unknown conditioning_type: {conditioning_type}")

		# Initialize parent with a placeholder task
		super().__init__(task=self.tasks[0], static_env_params=static_env_params)

	def reset_env(
		self,
		rng: jax.Array,
		params: EnvParams,
		task_id: int,  # Task ID is now essential for reset
		task_embeddings: jax.Array | None = None,
	) -> tuple[jax.Array, EnvState]:
		rng, world_rng = jax.random.split(rng)

		# 1. Generate the base world state for the task_id using lax.switch
		state = lax.switch(task_id, self.world_gen_fns, world_rng)

		# 2. Get the specific *clamped* TaskParams for this task_id from the stacked PyTree
		task_params = jax.tree.map(lambda x: x[task_id], self.stacked_task_params)

		# 3. Pre-populate achievements based on the initial state
		# This prevents getting rewards for items you spawn with
		state = calculate_inventory_achievements(state)

		achievements = state.achievements

		# Also pre-populate level-based achievements
		achievements = achievements.at[Achievement.ENTER_DUNGEON.value].set(state.player_level >= 1)
		achievements = achievements.at[Achievement.ENTER_GNOMISH_MINES.value].set(
			state.player_level >= 2
		)
		achievements = achievements.at[Achievement.ENTER_SEWERS.value].set(state.player_level >= 3)
		achievements = achievements.at[Achievement.ENTER_VAULT.value].set(state.player_level >= 4)
		achievements = achievements.at[Achievement.ENTER_TROLL_MINES.value].set(
			state.player_level >= 5
		)
		achievements = achievements.at[Achievement.ENTER_FIRE_REALM.value].set(
			state.player_level >= 6
		)
		achievements = achievements.at[Achievement.ENTER_ICE_REALM.value].set(
			state.player_level >= 7
		)
		achievements = achievements.at[Achievement.ENTER_GRAVEYARD.value].set(
			state.player_level >= 8
		)

		# 3. Store the task_id and the clamped task_params in the state
		state = state.replace(
			task_id=task_id,
			task_params=task_params,
			achievements=achievements,
		)

		return self.get_obs(state, task_embeddings), state

	def step_env(
		self,
		rng: jax.Array,
		state: EnvState,
		action: int,
		params: EnvParams,
		task_embeddings: jax.Array | None = None,
	) -> tuple[jax.Array, EnvState, float, bool, dict]:

		# was_success = lax.switch(state.task_id, self.success_fns, state)
		state, reward = self._craftax_step(rng, state, action, params)

		done = lax.switch(state.task_id, self.terminal_fns, state)
		success = lax.switch(state.task_id, self.success_fns, state)

		# 1. Get the pre-calculated intrinsic reward for this task
		base_task_reward = self.task_intrinsic_rewards[state.task_id]

		def get_static_bonus():
			scaled = base_task_reward * self.completion_bonus_scale
			return jnp.maximum(scaled, self.completion_bonus_min)

		def get_dynamic_bonus():
			scaled = state.running_original_return * self.dynamic_bonus_k
			return jnp.maximum(scaled, self.completion_bonus_min)

		if self.bonus_type == "static":
			final_bonus = get_static_bonus()
		elif self.bonus_type == "dynamic":
			final_bonus = get_dynamic_bonus()
		else:
			# Fallback or error
			final_bonus = get_static_bonus()

		# 2. Add the bonus *only if* the task was successfully completed
		success_bonus_reward = jax.lax.select(success, final_bonus, 0.0)

		reward += success_bonus_reward

		# reward = jnp.sign(reward) * jnp.log1p(jnp.abs(reward))

		info = log_achievements_to_info(state, done)
		info["discount"] = self.discount(state, params)
		info["is_success"] = success
		# Note: We could add task-specific terminal/reward logic here using lax.switch if needed

		obs = self.get_obs(state, task_embeddings)  # Call updated get_obs

		return (
			lax.stop_gradient(obs),
			lax.stop_gradient(state),
			reward,
			done,
			info,
		)

	def get_obs(
		self,
		state: EnvState,
		task_embeddings: jax.Array | None = None,  ### CHANGED ###
	) -> jax.Array:
		"""Returns the observation, conditionally concatenating a task vector.
		The agent is responsible for providing the embedding table if one is used.
		"""
		# symbolic_obs = render_craftax_symbolic(state)
		symbolic_obs = super().get_obs(state)

		if self.condition_on_task:
			# if self.conditioning_type == "one_hot":
			# 	task_vector = jax.nn.one_hot(state.task_id, num_classes=self.num_tasks)
			if self.conditioning_type in ["embedding", "one_hot"]:
				if task_embeddings is None:
					raise ValueError("task_embeddings must be provided for this conditioning type.")
				task_vector = task_embeddings[state.task_id]
			else:
				# This case is already handled in __init__, but included for completeness
				return symbolic_obs

			return jnp.concatenate([symbolic_obs, task_vector])
		else:
			return symbolic_obs

	def observation_space(self, params: EnvParams) -> spaces.Box:
		"""Returns the observation space, accounting for the concatenated task vector."""
		parent_space = super().observation_space(params)

		if self.condition_on_task:
			new_obs_shape = (parent_space.shape[0] + self.task_vector_size,)

			# Handle scalar bounds for the parent's observation space
			if jnp.isscalar(parent_space.low):
				low_bound = jnp.full(parent_space.shape, parent_space.low, dtype=parent_space.dtype)
			else:
				low_bound = parent_space.low

			if jnp.isscalar(parent_space.high):
				high_bound = jnp.full(
					parent_space.shape, parent_space.high, dtype=parent_space.dtype
				)
			else:
				high_bound = parent_space.high

			# Set bounds for the task vector part of the observation
			if self.conditioning_type == "one_hot":
				task_low = jnp.zeros(self.task_vector_size, dtype=parent_space.dtype)
				task_high = jnp.ones(self.task_vector_size, dtype=parent_space.dtype)

			### CHANGED ###
			# For embeddings, use -inf to inf for unbounded continuous values.
			# This is the most general and correct representation unless you
			# explicitly constrain them (e.g., with a tanh activation).
			elif self.conditioning_type == "embedding":
				task_low = jnp.full(self.task_vector_size, -1.0, dtype=parent_space.dtype)
				task_high = jnp.full(self.task_vector_size, 1.0, dtype=parent_space.dtype)
			else:
				raise ValueError(f"Unknown conditioning_type: {self.conditioning_type}")

			new_low = jnp.concatenate([low_bound, task_low])
			new_high = jnp.concatenate([high_bound, task_high])

			return spaces.Box(
				low=new_low, high=new_high, shape=new_obs_shape, dtype=parent_space.dtype
			)
		else:
			return parent_space


class MultiTaskMiniCraftaxEnvR(MiniCraftaxTrainR):
	"""A variant of MultiTaskMiniCraftaxEnv that uses task-specific rewards (R).

	Inherits from MiniCraftaxTrainR to utilize the `get_reward` method of tasks
	instead of the default achievement-based reward system.
	"""

	def __init__(
		self,
		task_classes: list[type[BaseTask]],
		static_env_params: StaticEnvParams,
		params: EnvParams,
		condition_on_task: bool = False,
		conditioning_type: str = "one_hot",
		embedding_size: int = 64,
		completion_bonus_scale: float = 2.0,
		completion_bonus_min: float = 20.0,
	):
		self.completion_bonus_scale = completion_bonus_scale
		self.completion_bonus_min = completion_bonus_min
		self.tasks = [task_cls(static_env_params, params) for task_cls in task_classes]
		self.num_tasks = len(self.tasks)

		# Store world gen functions (JAX-switchable)
		self.world_gen_fns = tuple(t.generate_world for t in self.tasks)
		self.reward_fns = tuple(t.get_reward for t in self.tasks)
		self.terminal_fns = tuple(t.is_terminal for t in self.tasks)
		self.success_fns = tuple(t.is_success for t in self.tasks)

		# --- Aggregate, CLAMP, and stack TaskParams ---
		self.all_task_params_raw = [t.get_task_params() for t in self.tasks]
		# Apply clamping during initialization
		self.all_task_params_clamped = [clamp_task_params(p) for p in self.all_task_params_raw]
		self.stacked_task_params = jax.tree.map(
			lambda *x: jnp.stack(x),
			*self.all_task_params_clamped,  # Stack the clamped parameters
		)
		# --- End TaskParams handling ---

		self.condition_on_task = condition_on_task
		self.conditioning_type = conditioning_type
		self.task_vector_size = embedding_size
		if condition_on_task:
			if conditioning_type == "one_hot":
				self.task_vector_size = self.num_tasks
			elif conditioning_type in ["embedding", "learned_embedding"]:
				self.task_vector_size = embedding_size
			else:
				raise ValueError(f"Unknown conditioning_type: {conditioning_type}")

		# Initialize parent with a placeholder task
		super().__init__(task=self.tasks[0], static_env_params=static_env_params)

	def reset_env(
		self,
		rng: jax.Array,
		params: EnvParams,
		task_id: int,  # Task ID is now essential for reset
		task_embeddings: jax.Array | None = None,
	) -> tuple[jax.Array, EnvState]:
		rng, world_rng = jax.random.split(rng)

		# 1. Generate the base world state for the task_id using lax.switch
		state = lax.switch(task_id, self.world_gen_fns, world_rng)

		# 2. Get the specific *clamped* TaskParams for this task_id from the stacked PyTree
		task_params = jax.tree.map(lambda x: x[task_id], self.stacked_task_params)

		# 3. Pre-populate achievements based on the initial state
		# This prevents getting rewards for items you spawn with
		state = calculate_inventory_achievements(state)

		achievements = state.achievements

		# Also pre-populate level-based achievements
		achievements = achievements.at[Achievement.ENTER_DUNGEON.value].set(state.player_level >= 1)
		achievements = achievements.at[Achievement.ENTER_GNOMISH_MINES.value].set(
			state.player_level >= 2
		)
		achievements = achievements.at[Achievement.ENTER_SEWERS.value].set(state.player_level >= 3)
		achievements = achievements.at[Achievement.ENTER_VAULT.value].set(state.player_level >= 4)
		achievements = achievements.at[Achievement.ENTER_TROLL_MINES.value].set(
			state.player_level >= 5
		)
		achievements = achievements.at[Achievement.ENTER_FIRE_REALM.value].set(
			state.player_level >= 6
		)
		achievements = achievements.at[Achievement.ENTER_ICE_REALM.value].set(
			state.player_level >= 7
		)
		achievements = achievements.at[Achievement.ENTER_GRAVEYARD.value].set(
			state.player_level >= 8
		)

		# 3. Store the task_id and the clamped task_params in the state
		state = state.replace(
			task_id=task_id,
			task_params=task_params,
			achievements=achievements,
		)

		return self.get_obs(state, task_embeddings), state

	def step_env(
		self,
		rng: jax.Array,
		state: EnvState,
		action: int,
		params: EnvParams,
		task_embeddings: jax.Array | None = None,
	) -> tuple[jax.Array, EnvState, float, bool, dict]:
		prev_state = state
		init_achievements = state.achievements
		init_health = state.player_health

		# Interrupt action if sleeping or resting
		action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)
		action = jax.lax.select(state.is_resting, Action.NOOP.value, action)

		# Change floor
		state = change_floor(state, action, params, self.static_env_params, state.task_params)

		# Crafting
		state = do_crafting(state, action)

		# Interact (mining, melee attacking, eating plants, drinking water)
		rng, _rng = jax.random.split(rng)
		state = do_action(_rng, state, action, self.static_env_params)

		# Placing
		state = place_block(state, action, self.static_env_params)

		# Shooting
		state = shoot_projectile(state, action, self.static_env_params)

		# Casting
		state = cast_spell(state, action, self.static_env_params)

		# Potions
		state = drink_potion(state, action)

		# Read
		rng, _rng = jax.random.split(rng)
		state = read_book(_rng, state, action)

		# Enchant
		rng, _rng = jax.random.split(rng)
		state = enchant(_rng, state, action)

		# Boss
		state = boss_logic(state, self.static_env_params)

		# Attributes
		state = level_up_attributes(state, action, params)

		# Movement
		state = move_player(state, action, params)

		# Mobs
		rng, _rng = jax.random.split(rng)
		state = update_mobs(_rng, state, params, self.static_env_params, state.task_params)

		rng, _rng = jax.random.split(rng)
		state = spawn_mobs(state, _rng, params, self.static_env_params, state.task_params)

		# Plants
		state = update_plants(state, self.static_env_params, state.task_params)

		# Intrinsics
		state = update_player_intrinsics(state, action, self.static_env_params, state.task_params)

		# Cap inv
		state = clip_inventory_and_intrinsics(state, params)

		# Inventory achievements
		state = calculate_inventory_achievements(state)

		achievement_reward = lax.switch(state.task_id, self.reward_fns, prev_state, state)
		reward = achievement_reward

		rng, _rng = jax.random.split(rng)

		state = state.replace(
			timestep=state.timestep + 1,
			light_level=calculate_light_level(state.timestep + 1, params),
			state_rng=_rng,
		)

		done = lax.switch(state.task_id, self.terminal_fns, state)
		success = lax.switch(state.task_id, self.success_fns, state)
		info = log_achievements_to_info(state, done)
		info["discount"] = self.discount(state, params)
		info["is_success"] = success
		# Note: We could add task-specific terminal/reward logic here using lax.switch if needed

		obs = self.get_obs(state, task_embeddings)  # Call updated get_obs

		return (
			lax.stop_gradient(obs),
			lax.stop_gradient(state),
			reward,
			done,
			info,
		)

	def get_obs(
		self,
		state: EnvState,
		task_embeddings: jax.Array | None = None,  ### CHANGED ###
	) -> jax.Array:
		"""Returns the observation, conditionally concatenating a task vector.
		The agent is responsible for providing the embedding table if one is used.
		"""
		symbolic_obs = render_craftax_symbolic(state)

		if self.condition_on_task:
			if self.conditioning_type == "one_hot":
				task_vector = jax.nn.one_hot(state.task_id, num_classes=self.num_tasks)
			elif self.conditioning_type in ["embedding", "learned_embedding"]:
				if task_embeddings is None:
					raise ValueError("task_embeddings must be provided for this conditioning type.")
				task_vector = task_embeddings[state.task_id]
			else:
				# This case is already handled in __init__, but included for completeness
				return symbolic_obs

			return jnp.concatenate([symbolic_obs, task_vector])
		else:
			return symbolic_obs

	def observation_space(self, params: EnvParams) -> spaces.Box:
		"""Returns the observation space, accounting for the concatenated task vector."""
		parent_space = super().observation_space(params)

		if self.condition_on_task:
			new_obs_shape = (parent_space.shape[0] + self.task_vector_size,)

			# Handle scalar bounds for the parent's observation space
			if jnp.isscalar(parent_space.low):
				low_bound = jnp.full(parent_space.shape, parent_space.low, dtype=parent_space.dtype)
			else:
				low_bound = parent_space.low

			if jnp.isscalar(parent_space.high):
				high_bound = jnp.full(
					parent_space.shape, parent_space.high, dtype=parent_space.dtype
				)
			else:
				high_bound = parent_space.high

			# Set bounds for the task vector part of the observation
			if self.conditioning_type == "one_hot":
				task_low = jnp.zeros(self.task_vector_size, dtype=parent_space.dtype)
				task_high = jnp.ones(self.task_vector_size, dtype=parent_space.dtype)

			### CHANGED ###
			# For embeddings, use -inf to inf for unbounded continuous values.
			# This is the most general and correct representation unless you
			# explicitly constrain them (e.g., with a tanh activation).
			elif self.conditioning_type in ["embedding", "learned_embedding"]:
				task_low = jnp.full(self.task_vector_size, -1.0, dtype=parent_space.dtype)
				task_high = jnp.full(self.task_vector_size, 1.0, dtype=parent_space.dtype)
			else:
				raise ValueError(f"Unknown conditioning_type: {self.conditioning_type}")

			new_low = jnp.concatenate([low_bound, task_low])
			new_high = jnp.concatenate([high_bound, task_high])

			return spaces.Box(
				low=new_low, high=new_high, shape=new_obs_shape, dtype=parent_space.dtype
			)
		else:
			return parent_space
