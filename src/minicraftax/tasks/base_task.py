import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement
from craftax.craftax.util.game_logic_utils import has_beaten_boss
from minicraftax.craftax_state import EnvState, TaskParams
from craftax.craftax.craftax_state import StaticEnvParams, EnvParams


class BaseTask(ABC):
	"""Abstract base class for all MiniCraftax tasks.

	Defines the interface for task-specific logic including world generation,
	reward calculation, and termination conditions.
	"""

	def __init__(self, static_env_params: StaticEnvParams, params: EnvParams):
		self.static_params = static_env_params
		self.params = params
		self.label = "base_task"
		self.relevant_achievements: list[Achievement] | None = None

	@abstractmethod
	def get_task_params(self) -> TaskParams:
		"""Returns the specific parameter configuration for this task."""
		pass

	def is_terminal(self, state) -> bool:
		"""Determines if the episode should end.

		The episode ends if:
		1. Max timesteps reached.
		2. Player dies.
		3. Boss is defeated.
		4. All relevant task achievements are completed.
		"""
		done_steps = state.timestep >= self.params.max_timesteps
		is_dead = state.player_health <= 0
		defeated_boss = has_beaten_boss(state, self.static_params)

		# 1. Get the boolean state of all achievements
		current_achievements_bool = state.achievements.astype(jnp.bool)

		# 2. Get the indices of the achievements we care about for this task
		relevant_indices = jnp.array([b.value for b in self.relevant_achievements])

		# 3. Check if all relevant achievements are True
		task_solved = jnp.all(current_achievements_bool[relevant_indices])

		return done_steps | is_dead | defeated_boss | task_solved

	def get_reward(self, prev_state, next_state):
		# This is the default logic for achievement-based tasks.
		achievement_coeff = ACHIEVEMENT_REWARD_MAP
		relevant_achievements = jnp.array([b.value for b in self.relevant_achievements])
		achievement_delta = next_state.achievements.astype(int) - prev_state.achievements.astype(
			int
		)
		mask = jnp.zeros_like(achievement_delta).at[relevant_achievements].set(1)
		achievement_reward = (achievement_delta * mask * achievement_coeff).sum()

		return achievement_reward

	@abstractmethod
	def generate_world(self, rng: jax.Array) -> EnvState:
		"""Generates the initial environment state for this task."""
		pass

	def is_success(self, state) -> bool:
		"""Returns True if the task's primary objective is met."""

		# 1. Get the boolean state of all achievements
		current_achievements_bool = state.achievements.astype(jnp.bool)

		# 2. Get the indices of the achievements we care about for this task
		relevant_indices = jnp.array([b.value for b in self.relevant_achievements])

		# 3. Check if all relevant achievements are True
		task_solved = jnp.all(current_achievements_bool[relevant_indices])

		return task_solved


def clamp_task_params(task_params: TaskParams) -> TaskParams:
	"""Clamps task parameters to safe/reasonable ranges.

	Ensures multipliers are non-negative and integer values are within valid bounds.
	"""
	return task_params.replace(
		passive_spawn_multiplier=jnp.maximum(task_params.passive_spawn_multiplier, 0.0),
		melee_spawn_multiplier=jnp.maximum(task_params.melee_spawn_multiplier, 0.0),
		ranged_spawn_multiplier=jnp.maximum(task_params.ranged_spawn_multiplier, 0.0),
		mob_health_multiplier=jnp.maximum(task_params.mob_health_multiplier, 0.01),
		mob_damage_multiplier=jnp.maximum(task_params.mob_damage_multiplier, 0.0),
		melee_trigger_distance=jnp.maximum(task_params.melee_trigger_distance, 1),
		monsters_killed_to_clear_level=jnp.maximum(task_params.monsters_killed_to_clear_level, 0),
		needs_depletion_multiplier=jnp.maximum(task_params.needs_depletion_multiplier, 0.0),
		health_recover_multiplier=jnp.maximum(task_params.health_recover_multiplier, 0.01),
		health_loss_multiplier=jnp.maximum(task_params.health_loss_multiplier, 0.0),
		mana_recover_multiplier=jnp.maximum(task_params.mana_recover_multiplier, 0.01),
		growing_plants_age=jnp.maximum(task_params.growing_plants_age, 2),
	)
