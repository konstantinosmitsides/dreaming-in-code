import jax
import jax.numpy as jnp
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement, Action
from craftax.craftax.craftax_state import EnvParams, EnvState, StaticEnvParams
from craftax.craftax.envs.common import log_achievements_to_info
from craftax.craftax.envs.craftax_symbolic_env import (
	CraftaxSymbolicEnv,
	CraftaxSymbolicEnvNoAutoReset,
)
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
from craftax.craftax.util.game_logic_utils import calculate_light_level
from jax import lax

from minicraftax.game_mechanics import (
	change_floor,
	spawn_mobs,
	update_mobs,
	update_plants,
	update_player_intrinsics,
)
from minicraftax.tasks.base_task import BaseTask


class MiniCraftaxTrain(CraftaxSymbolicEnvNoAutoReset):
	"""A wrapper for the MiniCraftax environment optimized for training.

	This class handles the interaction loop, logging achievements, and managing state
	transitions without automatic resets, allowing for explicit reset control.

	Attributes:
		task (BaseTask): The specific task instance defining the goal and rewards.
		label (str): Label of the task.
	"""

	def __init__(self, task: BaseTask, static_env_params: StaticEnvParams | None = None):
		self.task = task
		super().__init__(static_env_params=static_env_params)
		self.label = task.label

	def step_env(
		self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
	) -> tuple[jax.Array, EnvState, float, bool, dict]:
		# state, reward = craftax_step(rng, state, action, params, self.static_env_params)
		state, reward = self._craftax_step(rng, state, action, params)

		done = self.task.is_terminal(state)
		info = log_achievements_to_info(state, done)
		info["discount"] = self.discount(state, params)

		return (
			lax.stop_gradient(self.get_obs(state)),
			lax.stop_gradient(state),
			reward,
			done,
			info,
		)

	def reset_env(self, rng: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
		rng, _rng = jax.random.split(rng)
		state = self.task.generate_world(_rng)

		# current_achievements = state.achievements.astype(int)
		# achievements = jnp.zeros_like(current_achievements)

		# Pre-populate achievements based on the initial state
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

		# Get the task-specific parameters from the task object
		task_params = self.task.get_task_params()

		# Store the retrieved parameters in the state object
		state = state.replace(
			task_params=task_params,
			achievements=achievements,
		)

		return self.get_obs(state), state

	def _craftax_step(
		self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
	) -> tuple[EnvState, float]:
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

		# Reward
		achievement_coefficients = ACHIEVEMENT_REWARD_MAP
		achievement_reward = (
			(state.achievements.astype(int) - init_achievements.astype(int))
			* achievement_coefficients
		).sum()
		health_reward = (state.player_health - init_health) * 0.1

		reward = achievement_reward + health_reward

		rng, _rng = jax.random.split(rng)

		state = state.replace(
			timestep=state.timestep + 1,
			light_level=calculate_light_level(state.timestep + 1, params),
			state_rng=_rng,
		)

		return state, reward


class MiniCraftaxEval(CraftaxSymbolicEnv):
	"""A wrapper for the MiniCraftax environment optimized for evaluation.

	Similar to MiniCraftaxTrain but purely for evaluation purposes, adhering to the
	standard CraftaxSymbolicEnv interface.

	Attributes:
		task (BaseTask): The specific task instance.
	"""

	def __init__(self, task: BaseTask, static_env_params: StaticEnvParams | None = None):
		self.task = task
		super().__init__(static_env_params=static_env_params)

	def step_env(
		self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
	) -> tuple[jax.Array, EnvState, float, bool, dict]:
		# state, reward = craftax_step(rng, state, action, params, self.static_env_params)
		state, reward = self._craftax_step(rng, state, action, params)

		done = self.task.is_terminal(state)
		info = log_achievements_to_info(state, done)
		info["discount"] = self.discount(state, params)

		return (
			lax.stop_gradient(self.get_obs(state)),
			lax.stop_gradient(state),
			reward,
			done,
			info,
		)

	def reset_env(self, rng: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
		rng, _rng = jax.random.split(rng)
		state = self.task.generate_world(_rng)

		# current_achievements = state.achievements.astype(int)
		# achievements = jnp.zeros_like(current_achievements)

		# Pre-populate achievements based on the initial state
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

		# Get the task-specific parameters from the task object
		task_params = self.task.get_task_params()

		# Store the retrieved parameters in the state object
		state = state.replace(
			task_params=task_params,
			achievements=achievements,
		)

		return self.get_obs(state), state

	def _craftax_step(
		self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
	) -> tuple[EnvState, float]:
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

		# Reward calculation: sum of achievement deltas and health changes.
		# Note: Uses hardcoded coefficients from ACHIEVEMENT_REWARD_MAP.
		achievement_coefficients = ACHIEVEMENT_REWARD_MAP
		achievement_reward = (
			(state.achievements.astype(int) - init_achievements.astype(int))
			* achievement_coefficients
		).sum()
		health_reward = (state.player_health - init_health) * 0.1
		reward = achievement_reward + health_reward

		rng, _rng = jax.random.split(rng)

		state = state.replace(
			timestep=state.timestep + 1,
			light_level=calculate_light_level(state.timestep + 1, params),
			state_rng=_rng,
		)

		return state, reward


class MiniCraftaxTrainR(CraftaxSymbolicEnvNoAutoReset):
	"""A variant of MiniCraftaxTrain that uses task-specific rewards.

	Instead of global achievement rewards, this class delegates reward calculation
	to the `get_reward` method of the specific task.
	"""

	def __init__(self, task: BaseTask, static_env_params: StaticEnvParams | None = None):
		self.task = task
		super().__init__(static_env_params=static_env_params)

	def step_env(
		self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
	) -> tuple[jax.Array, EnvState, float, bool, dict]:
		# state, reward = craftax_step(rng, state, action, params, self.static_env_params)
		state, reward = self._craftax_step(rng, state, action, params)

		done = self.task.is_terminal(state)
		info = log_achievements_to_info(state, done)
		info["discount"] = self.discount(state, params)

		return (
			lax.stop_gradient(self.get_obs(state)),
			lax.stop_gradient(state),
			reward,
			done,
			info,
		)

	def reset_env(self, rng: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
		rng, _rng = jax.random.split(rng)
		state = self.task.generate_world(_rng)

		# current_achievements = state.achievements.astype(int)
		# achievements = jnp.zeros_like(current_achievements)

		# Pre-populate achievements based on the initial state
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

		# Get the task-specific parameters from the task object
		task_params = self.task.get_task_params()

		# Store the retrieved parameters in the state object
		state = state.replace(
			task_params=task_params,
			achievements=achievements,
		)

		return self.get_obs(state), state

	def _craftax_step(
		self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
	) -> tuple[EnvState, float]:
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

		# Reward
		# achievement_coefficients = ACHIEVEMENT_REWARD_MAP
		# achievement_reward = (
		#     (state.achievements.astype(int) - init_achievements.astype(int))
		#     * achievement_coefficients
		# ).sum()
		health_reward = (state.player_health - init_health) * 0.1
		# reward = achievement_reward + health_reward
		achievement_reward = self.task.get_reward(prev_state, state)
		reward = achievement_reward + health_reward

		rng, _rng = jax.random.split(rng)

		state = state.replace(
			timestep=state.timestep + 1,
			light_level=calculate_light_level(state.timestep + 1, params),
			state_rng=_rng,
		)

		return state, reward


class MiniCraftaxEvalR(CraftaxSymbolicEnv):
	"""A variant of MiniCraftaxEval that uses task-specific rewards.

	Useful for evaluation scenarios where the metric of interest is defined by the task's
	reward function rather than standard game achievements.
	"""

	def __init__(self, task: BaseTask, static_env_params: StaticEnvParams | None = None):
		self.task = task
		super().__init__(static_env_params=static_env_params)

	def step_env(
		self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
	) -> tuple[jax.Array, EnvState, float, bool, dict]:
		# state, reward = craftax_step(rng, state, action, params, self.static_env_params)
		state, reward = self._craftax_step(rng, state, action, params)

		done = self.task.is_terminal(state)
		info = log_achievements_to_info(state, done)
		info["discount"] = self.discount(state, params)

		return (
			lax.stop_gradient(self.get_obs(state)),
			lax.stop_gradient(state),
			reward,
			done,
			info,
		)

	def reset_env(self, rng: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:
		rng, _rng = jax.random.split(rng)
		state = self.task.generate_world(_rng)

		# current_achievements = state.achievements.astype(int)
		# achievements = jnp.zeros_like(current_achievements)

		# Pre-populate achievements based on the initial state
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

		# Get the task-specific parameters from the task object
		task_params = self.task.get_task_params()

		# Store the retrieved parameters in the state object
		state = state.replace(
			task_params=task_params,
			achievements=achievements,
		)

		return self.get_obs(state), state

	def _craftax_step(
		self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
	) -> tuple[EnvState, float]:
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

		# Reward
		# achievement_coefficients = ACHIEVEMENT_REWARD_MAP
		# achievement_reward = (
		#     (state.achievements.astype(int) - init_achievements.astype(int))
		#     * achievement_coefficients
		# ).sum()
		health_reward = (state.player_health - init_health) * 0.1
		# reward = achievement_reward + health_reward
		achievement_reward = self.task.get_reward(prev_state, state)
		reward = achievement_reward + health_reward

		rng, _rng = jax.random.split(rng)

		state = state.replace(
			timestep=state.timestep + 1,
			light_level=calculate_light_level(state.timestep + 1, params),
			state_rng=_rng,
		)

		return state, reward
