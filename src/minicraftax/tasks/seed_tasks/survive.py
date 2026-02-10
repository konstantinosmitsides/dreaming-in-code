import jax
from craftax.craftax.constants import Achievement, BlockType
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
	"""Objective: Manage all survival needs (hunger, thirst, and energy).
	Description: The player must achieve `EAT_COW`, `COLLECT_DRINK`, `WAKE_UP`. The world is a standard procedural overworld with 3 cows (4-8 tiles away). All mob spawning is enabled with  very easy settings, and all player needs are enabled with default depletion rates.
	Relevant Achievements: EAT_COW, COLLECT_DRINK, WAKE_UP
	Completed Achievements: None
	World:
	- Player: Starts on floor 0 with an empty inventory.
	- Map: 3 `COW` (passive mob type_id=0) are randomly placed 4-8 tiles away.
	- Mechanics: "needs_depletion_multiplier = 1.0", "passive_spawn_multiplier = 1.0", "melee_spawn_multiplier = 0.05", "ranged_spawn_multiplier = 0.05"
	"""

	def __init__(self, static_params: StaticEnvParams, params: EnvParams):
		super().__init__(static_params, params)
		# We now check for all survival achievements
		self.relevant_achievements = [
			Achievement.EAT_COW,
			Achievement.COLLECT_DRINK,
			Achievement.WAKE_UP,
		]
		self.completed_achievements = []
		self.label = "EAT_COW, COLLECT_DRINK, WAKE_UP"

	def get_task_params(self) -> TaskParams:
		"""Return custom parameters for this task."""
		return TaskParams(
			passive_spawn_multiplier=1.0,
			melee_spawn_multiplier=0.05,
			ranged_spawn_multiplier=0.05,
			needs_depletion_multiplier=1.0,  # Enables all needs
		)

	def generate_world(self, rng: jax.Array) -> EnvState:
		"""Generates the world for the task."""
		rng, build_rng, cow_rng = jax.random.split(rng, 3)

		builder = WorldBuilder(build_rng, self.static_params, self.params)

		builder.set_starting_floor(0)

		# Place 3 cows near the player on level 0
		builder.add_mobs_randomly_near(
			cow_rng,
			level=0,
			mob_name="passive",
			type_id=0,
			n=3,
			target_pos=builder.player_position,
			min_dist=4,
			max_dist=8,
			on_blocks=[BlockType.GRASS, BlockType.PATH, BlockType.SAND],
		)

		return builder.build(rng)
