import jax
from craftax.craftax.constants import Achievement, BlockType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
	"""Objective: Defeat a zombie when you have a wooden sword and you recover 5 times faster.
	Description: The player must achieve the `DEFEAT_ZOMBIE` achievement. The player starts on Floor 0 (the overworld) with a wooden sword and nearby cows. One zombie is placed 4-8 tiles from the player's start. All player needs are enabled, and passive and melee mobs are enabled in case the starting ones despawn. Ranged mobs are enabled with easier settings.
	Relevant Achievements: DEFEAT_ZOMBIE
	Completed Achievements: MAKE_WOOD_SWORD
	World:
	- Player: Starts on floor 0 with a wooden sword (`{"sword": 1}`)
	- Map: One `ZOMBIE` (melee mob type_id=0) and 3 `COW` (passive mob type_id=0) are placed randomly within 4-8 (Manhattan distance) tiles of the player.
	- Mechanics: "needs_depletion_multiplier = 1.0", "passive_spawn_multiplier = 1.0", "melee_spawn_multiplier = 1.0", "ranged_spawn_multiplier = 0.2, health_recover_multiplier = 5.0"
	"""

	def __init__(self, static_params: StaticEnvParams, params: EnvParams):
		super().__init__(static_params, params)
		self.relevant_achievements = [Achievement.DEFEAT_ZOMBIE]
		self.completed_achievements = [Achievement.MAKE_WOOD_SWORD]
		self.label = "DEFEAT_ZOMBIE"

	def get_task_params(self) -> TaskParams:
		"""Return custom parameters for this task."""
		return TaskParams(
			passive_spawn_multiplier=1.0,
			melee_spawn_multiplier=1.0,
			ranged_spawn_multiplier=0.2,
			needs_depletion_multiplier=1.0,  # Needs are ON
			health_recover_multiplier=5.0,  # Keep high regen
		)

	def generate_world(self, rng: jax.Array) -> EnvState:
		"""Generates the world for the task."""
		rng, build_rng, mob_rng, cow_rng = jax.random.split(rng, 4)

		builder = WorldBuilder(build_rng, self.static_params, self.params)

		builder.set_starting_floor(0)
		builder.set_player_inventory({"sword": 1})  # 1 = wood sword

		# Place 1 zombie near the player on level 0
		builder.add_mobs_randomly_near(
			mob_rng,
			level=0,
			mob_name="melee",
			type_id=0,  # type_id 0 is Zombie
			n=1,
			target_pos=builder.player_position,
			min_dist=4,
			max_dist=8,
			on_blocks=[BlockType.GRASS, BlockType.PATH, BlockType.SAND],
		)

		# --- ADDED SCAFFOLDING ---
		# 2. Place cows as a food source
		builder.add_mobs_randomly_near(
			cow_rng,
			level=0,
			mob_name="passive",
			type_id=0,  # type_id 0 is Cow
			n=3,
			target_pos=builder.player_position,
			min_dist=4,
			max_dist=8,
			on_blocks=[BlockType.GRASS, BlockType.PATH],
		)
		# --- END SCAFFOLDING ---

		return builder.build(rng)
