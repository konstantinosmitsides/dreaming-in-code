import jax
from craftax.craftax.constants import Achievement, BlockType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
	"""Objective: Collect coal.
	Description: The player must achieve the `COLLECT_COAL` achievement. The player starts on Floor 0 (the overworld) with a wooden pickaxe and sword. The world is a standard procedural overworld with 5 coal blocks placed 4-8 tiles from the player's start. Mobs and needs are enabled but with easier settings.
	Relevant Achievements: COLLECT_COAL
	Completed Achievements: MAKE_WOOD_PICKAXE, MAKE_WOOD_SWORD
	World:
	- Player: Starts on floor 0 with a wooden pickaxe and wooden sword (`{"pickaxe": 1, "sword": 1}`).
	- Map: 5 `COAL` blocks are placed randomly on `GRASS` or `STONE` within 4-8 (Manhattan distance) tiles of the player. 3 `COW` (passive mob type_id=0) are placed 4-8 tiles away.
	- Mechanics: "needs_depletion_multiplier = 0.5", "passive_spawn_multiplier = 1.0", "melee_spawn_multiplier = 0.2", "ranged_spawn_multiplier = 0.2"
	"""

	def __init__(self, static_params: StaticEnvParams, params: EnvParams):
		super().__init__(static_params, params)
		self.relevant_achievements = [Achievement.COLLECT_COAL]
		self.completed_achievements = [Achievement.MAKE_WOOD_PICKAXE, Achievement.MAKE_WOOD_SWORD]
		self.label = "COLLECT_COAL"

	def get_task_params(self) -> TaskParams:
		"""Return custom parameters for this task."""
		return TaskParams(
			passive_spawn_multiplier=1.0,  # Enable random cow spawns
			melee_spawn_multiplier=0.2,  # Enable zombie spawns
			ranged_spawn_multiplier=0.2,  # Enable skeleton spawns
			needs_depletion_multiplier=0.5,  # Needs are on, but slow
		)

	def generate_world(self, rng: jax.Array) -> EnvState:
		"""Generates the world for the task."""
		rng, build_rng, placement_rng, cow_rng = jax.random.split(rng, 4)

		builder = WorldBuilder(build_rng, self.static_params, self.params)

		builder.set_starting_floor(0)

		# --- ADDED SCAFFOLDING ---
		# 1. Give prerequisite pickaxe and a sword for safety
		builder.set_player_inventory({"pickaxe": 1, "sword": 1})

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

		# Place 5 coal blocks near the player on level 0
		builder.place_randomly_near(
			placement_rng,
			level=0,
			block_type=BlockType.COAL,
			target_pos=builder.player_position,
			min_dist=4,
			max_dist=8,
			n=5,
			on_blocks=[BlockType.GRASS, BlockType.STONE],
		)

		return builder.build(rng)
