import gymnax.environments.spaces as spaces
import jax
import jax.numpy as jnp
from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, EnvState, StaticEnvParams
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset


class CraftaxAugObsTrain(CraftaxSymbolicEnvNoAutoReset):
	def __init__(
		self,
		static_env_params: StaticEnvParams = None,
		condition_on_task: bool = False,
		conditioning_type: str = "one_hot",
		num_tasks: int = 1,
		embedding_size: int = 67,
		task_embeddings: jax.Array | None = None,
	):
		super().__init__(static_env_params=static_env_params)

		self.condition_on_task = condition_on_task
		self.conditioning_type = conditioning_type
		self.num_tasks = num_tasks
		self.embedding_size = embedding_size
		# self.task_vector_size = 0

		self.relevant_achievements = [
			Achievement.COLLECT_WOOD,
			Achievement.PLACE_TABLE,
			Achievement.EAT_COW,
			Achievement.COLLECT_SAPLING,
			Achievement.COLLECT_DRINK,
			Achievement.MAKE_WOOD_PICKAXE,
			Achievement.MAKE_WOOD_SWORD,
			Achievement.PLACE_PLANT,
			Achievement.DEFEAT_ZOMBIE,
			Achievement.COLLECT_STONE,
			Achievement.PLACE_STONE,
			Achievement.EAT_PLANT,
			Achievement.DEFEAT_SKELETON,
			Achievement.MAKE_STONE_PICKAXE,
			Achievement.MAKE_STONE_SWORD,
			Achievement.WAKE_UP,
			Achievement.PLACE_FURNACE,
			Achievement.COLLECT_COAL,
			Achievement.COLLECT_IRON,
			Achievement.COLLECT_DIAMOND,
			Achievement.MAKE_IRON_PICKAXE,
			Achievement.MAKE_IRON_SWORD,
			Achievement.MAKE_ARROW,
			Achievement.MAKE_TORCH,
			Achievement.PLACE_TORCH,
			Achievement.COLLECT_SAPPHIRE,
			Achievement.COLLECT_RUBY,
			Achievement.MAKE_DIAMOND_PICKAXE,
			Achievement.MAKE_DIAMOND_SWORD,
			Achievement.MAKE_IRON_ARMOUR,
			Achievement.MAKE_DIAMOND_ARMOUR,
			Achievement.ENTER_GNOMISH_MINES,
			Achievement.ENTER_DUNGEON,
			Achievement.ENTER_SEWERS,
			Achievement.ENTER_VAULT,
			Achievement.ENTER_TROLL_MINES,
			Achievement.ENTER_FIRE_REALM,
			Achievement.ENTER_ICE_REALM,
			Achievement.ENTER_GRAVEYARD,
			Achievement.DEFEAT_GNOME_WARRIOR,
			Achievement.DEFEAT_GNOME_ARCHER,
			Achievement.DEFEAT_ORC_SOLIDER,
			Achievement.DEFEAT_ORC_MAGE,
			Achievement.DEFEAT_LIZARD,
			Achievement.DEFEAT_KOBOLD,
			Achievement.DEFEAT_KNIGHT,
			Achievement.DEFEAT_ARCHER,
			Achievement.DEFEAT_TROLL,
			Achievement.DEFEAT_DEEP_THING,
			Achievement.DEFEAT_PIGMAN,
			Achievement.DEFEAT_FIRE_ELEMENTAL,
			Achievement.DEFEAT_FROST_TROLL,
			Achievement.DEFEAT_ICE_ELEMENTAL,
			Achievement.DAMAGE_NECROMANCER,
			Achievement.DEFEAT_NECROMANCER,
			Achievement.EAT_BAT,
			Achievement.EAT_SNAIL,
			Achievement.FIND_BOW,
			Achievement.FIRE_BOW,
			Achievement.LEARN_FIREBALL,
			Achievement.CAST_FIREBALL,
			Achievement.LEARN_ICEBALL,
			Achievement.CAST_ICEBALL,
			Achievement.OPEN_CHEST,
			Achievement.DRINK_POTION,
			Achievement.ENCHANT_SWORD,
			Achievement.ENCHANT_ARMOUR,
		]

		self.label = "COLLECT_WOOD, PLACE_TABLE, EAT_COW, COLLECT_SAPLING, COLLECT_DRINK, \
        MAKE_WOOD_PICKAXE, MAKE_WOOD_SWORD, PLACE_PLANT, DEFEAT_ZOMBIE, COLLECT_STONE, PLACE_STONE, EAT_PLANT, \
        DEFEAT_SKELETON, MAKE_STONE_PICKAXE, MAKE_STONE_SWORD, WAKE_UP, PLACE_FURNACE, COLLECT_COAL, COLLECT_IRON, \
        COLLECT_DIAMOND, MAKE_IRON_PICKAXE, MAKE_IRON_SWORD, MAKE_ARROW, MAKE_TORCH, PLACE_TORCH, COLLECT_SAPPHIRE, \
        COLLECT_RUBY, MAKE_DIAMOND_PICKAXE, MAKE_DIAMOND_SWORD, MAKE_IRON_ARMOUR, MAKE_DIAMOND_ARMOUR, ENTER_GNOMISH_MINES, \
        ENTER_DUNGEON, ENTER_SEWERS, ENTER_VAULT, ENTER_TROLL_MINES, ENTER_FIRE_REALM, ENTER_ICE_REALM, ENTER_GRAVEYARD, \
        DEFEAT_GNOME_WARRIOR, DEFEAT_GNOME_ARCHER, DEFEAT_ORC_SOLIDER, DEFEAT_ORC_MAGE, DEFEAT_LIZARD, DEFEAT_KOBOLD, \
        DEFEAT_KNIGHT, DEFEAT_ARCHER, DEFEAT_TROLL, DEFEAT_DEEP_THING, DEFEAT_PIGMAN, DEFEAT_FIRE_ELEMENTAL, \
        DEFEAT_FROST_TROLL, DEFEAT_ICE_ELEMENTAL, DAMAGE_NECROMANCER, DEFEAT_NECROMANCER, EAT_BAT, EAT_SNAIL, FIND_BOW, \
        FIRE_BOW, LEARN_FIREBALL, CAST_FIREBALL, LEARN_ICEBALL, CAST_ICEBALL, OPEN_CHEST, DRINK_POTION, ENCHANT_SWORD, \
        ENCHANT_ARMOUR"

		if condition_on_task:
			# if conditioning_type == "one_hot":
			# 	self.task_vector_size = self.num_tasks
			if conditioning_type in ["embedding", "one_hot"]:
				if task_embeddings is None:
					raise ValueError(
						"task_embeddings table must be provided to __init__ "
						"for this conditioning type."
					)
				# self.task_vector_size = self.embedding_size
				self.task_embeddings = task_embeddings
			else:
				raise ValueError(f"Unknown conditioning_type: {conditioning_type}")

	def get_obs(
		self,
		state: EnvState,
	) -> jax.Array:
		"""Returns the observation, conditionally concatenating a task vector.
		The agent is responsible for providing the embedding table if one is used.
		"""
		symbolic_obs = super().get_obs(state)

		if self.condition_on_task:
			# if self.conditioning_type == "one_hot":
			# 	task_vector = jax.nn.one_hot(0, num_classes=self.task_vector_size)
			if self.conditioning_type in ["embedding", "one_hot"]:
				task_vector = self.task_embeddings[0]
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
			new_obs_shape = (parent_space.shape[0] + self.embedding_size,)

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
				task_low = jnp.zeros(self.embedding_size, dtype=parent_space.dtype)
				task_high = jnp.ones(self.embedding_size, dtype=parent_space.dtype)

			### CHANGED ###
			# For embeddings, use -inf to inf for unbounded continuous values.
			# This is the most general and correct representation unless you
			# explicitly constrain them (e.g., with a tanh activation).
			elif self.conditioning_type == "embedding":
				task_low = jnp.full(self.embedding_size, -1.0, dtype=parent_space.dtype)
				task_high = jnp.full(self.embedding_size, 1.0, dtype=parent_space.dtype)
			else:
				raise ValueError(f"Unknown conditioning_type: {self.conditioning_type}")

			new_low = jnp.concatenate([low_bound, task_low])
			new_high = jnp.concatenate([high_bound, task_high])

			return spaces.Box(
				low=new_low, high=new_high, shape=new_obs_shape, dtype=parent_space.dtype
			)
		else:
			return parent_space
