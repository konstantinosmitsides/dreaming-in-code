import jax
from craftax.craftax.constants import Achievement, BlockType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
	"""Objective: Craft a wooden pickaxe.
	Description: The player must achieve `COLLECT_WOOD`, `PLACE_TABLE`, and `MAKE_WOOD_PICKAXE`. The player starts on Floor 0 (the overworld) with a wooden sword for safety and nearby cows for food. Mobs and survival needs are enabled to encourage opportunistic learning but with easier settings - Melee and Ranged Mobs rates are extremely low.
	Relevant Achievements: COLLECT_WOOD, PLACE_TABLE, MAKE_WOOD_PICKAXE
	Completed Achievements: MAKE_WOOD_SWORD
	World:
	- Player: Starts on floor 0 with a wooden sword (`{"sword": 1}`).
	- Map: Default procedural overworld (Floor 0). 3 `COW` mobs (passive mob type_id=0) are placed 4-8 tiles from the player.
	- Mechanics: "needs_depletion_multiplier = 0.5", "passive_spawn_multiplier = 1.0", "melee_spawn_multiplier = 0.1", "ranged_spawn_multiplier = 0.05"
	"""

	def __init__(self, static_params: StaticEnvParams, params: EnvParams):
		super().__init__(static_params, params)
		self.relevant_achievements = [
			Achievement.COLLECT_WOOD,
			Achievement.PLACE_TABLE,
			Achievement.MAKE_WOOD_PICKAXE,
		]
		self.completed_achievements = [Achievement.MAKE_WOOD_SWORD]
		self.label = "COLLECT_WOOD, PLACE_TABLE, MAKE_WOOD_PICKAXE"

	def get_task_params(self) -> TaskParams:
		"""Return custom parameters for this task."""
		return TaskParams(
			passive_spawn_multiplier=1.0,  # Enable random cow spawns
			melee_spawn_multiplier=0.1,  # Enable zombie spawns
			ranged_spawn_multiplier=0.05,  # Enable skeleton spawns
			needs_depletion_multiplier=0.5,  # Needs are on, but slow
		)

	def generate_world(self, rng: jax.Array) -> EnvState:
		"""Generates the world for the task."""
		rng, build_rng, cow_rng = jax.random.split(rng, 3)

		builder = WorldBuilder(build_rng, self.static_params, self.params)

		builder.set_starting_floor(0)

		# --- ADDED SCAFFOLDING ---
		# 1. Give a sword for safety
		builder.set_player_inventory({"sword": 1})  # 1 = wood sword

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
