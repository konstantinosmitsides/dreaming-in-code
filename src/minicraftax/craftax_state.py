from typing import Any

import jax.numpy as jnp
from craftax.craftax.craftax_state import Inventory, Mobs
from flax import struct


@struct.dataclass
class TaskParams:
    """Holds parameters that customize the MiniCraftax task.

    Attributes:
        passive_spawn_multiplier: Multiplier for passive mob spawn chance.
        melee_spawn_multiplier: Multiplier for melee mob spawn chance.
        ranged_spawn_multiplier: Multiplier for ranged mob spawn chance.
        mob_health_multiplier: Multiplier for mob base health.
        mob_damage_multiplier: Multiplier for mob base damage.
        melee_trigger_distance: Distance at which melee mobs start chasing player.
        monsters_killed_to_clear_level: Monsters needed to unlock next level ladder.
        needs_depletion_multiplier: Multiplier for hunger/thirst/fatigue rates.
        health_recover_multiplier: Multiplier for health recovery rate.
        health_loss_multiplier: Multiplier for health loss rate (when needs unmet).
        mana_recover_multiplier: Multiplier for mana recovery rate.
        growing_plants_age: Timesteps for a plant to become ripe.
    """

    passive_spawn_multiplier: float = 1.0       # Multiplier for passive mob spawn chance
    melee_spawn_multiplier: float = 1.0         # Multiplier for melee mob spawn chance
    ranged_spawn_multiplier: float = 1.0        # Multiplier for ranged mob spawn chance
    mob_health_multiplier: float = 1.0          # Multiplier for mob base health
    mob_damage_multiplier: float = 1.0          # Multiplier for mob base damage
    melee_trigger_distance: int = 10            # Distance at which melee mobs start chasing player
    monsters_killed_to_clear_level: int = 8     # Monsters needed to unlock next level ladder
    needs_depletion_multiplier: float = 1.0     # Multiplier for hunger/thirst/fatigue rates
    health_recover_multiplier: float = 1.0      # Multiplier for health recovery rate
    health_loss_multiplier: float = 1.0         # Multiplier for health loss rate (when needs unmet)
    mana_recover_multiplier: float = 1.0        # Multiplier for mana recovery rate
    growing_plants_age: int = 600               # Timesteps for a plant to become ripe


@struct.dataclass
class EnvState:
	"""Represents the full state of the MiniCraftax environment.

	Includes the map, player state, mob states, and other game mechanics data.
	"""
	task_id: int
	map: jnp.ndarray
	item_map: jnp.ndarray
	mob_map: jnp.ndarray
	light_map: jnp.ndarray
	down_ladders: jnp.ndarray
	up_ladders: jnp.ndarray
	chests_opened: jnp.ndarray
	monsters_killed: jnp.ndarray

	player_position: jnp.ndarray
	player_level: int
	player_direction: int

	# Intrinsics
	player_health: float
	player_food: int
	player_drink: int
	player_energy: int
	player_mana: int
	is_sleeping: bool
	is_resting: bool

	# Second order intrinsics
	player_recover: float
	player_hunger: float
	player_thirst: float
	player_fatigue: float
	player_recover_mana: float

	# Attributes
	player_xp: int
	player_dexterity: int
	player_strength: int
	player_intelligence: int

	inventory: Inventory

	melee_mobs: Mobs
	passive_mobs: Mobs
	ranged_mobs: Mobs

	mob_projectiles: Mobs
	mob_projectile_directions: jnp.ndarray
	player_projectiles: Mobs
	player_projectile_directions: jnp.ndarray

	growing_plants_positions: jnp.ndarray
	growing_plants_age: jnp.ndarray
	growing_plants_mask: jnp.ndarray

	potion_mapping: jnp.ndarray
	learned_spells: jnp.ndarray

	sword_enchantment: int
	bow_enchantment: int
	armour_enchantments: jnp.ndarray

	boss_progress: int
	boss_timesteps_to_spawn_this_round: int

	light_level: float

	achievements: jnp.ndarray

	state_rng: Any

	timestep: int

	fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)

	# NEW: Store the running average return of the original task
	running_original_return: float = 0.0

	# We use a default_factory because TaskParams() creates a concrete instance,
	# which can cause issues with JAX transformations if defined directly as a default.
	# The factory ensures a fresh instance is created when needed during initialization.
	task_params: TaskParams = struct.field(pytree_node=True, default_factory=TaskParams)
