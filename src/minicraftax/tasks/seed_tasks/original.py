import jax
from craftax.craftax.constants import Achievement

from minicraftax.craftax_state import EnvState, TaskParams
from minicraftax.tasks.base_task import BaseTask
from minicraftax.world_builder import WorldBuilder


class Env(BaseTask):
	"""Description: This is the original Craftax task.
	Changes in TaskParams:
	- passive_spawn_multiplier: 1.0
	- melee_spawn_multiplier: 1.0
	- ranged_spawn_multiplier: 1.0
	- mob_health_multiplier: 1.0
	- mob_damage_multiplier: 1.0
	- melee_trigger_distance: 10
	- monsters_killed_to_clear_level: 8
	- needs_depletion_multiplier: 1.0
	- health_recover_multiplier: 1.0
	- health_loss_multiplier: 1.0
	- mana_recover_multiplier: 1.0
	- growing_plants_age: 600
	Changes in World Generation: Default Generation.
	Terminal Condition: ENTER_DUNGEON
	"""

	def __init__(self, static_params, params):
		super().__init__(static_params, params)
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

		self.completed_achievements = []

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

	def get_task_params(self) -> TaskParams:
		"""Return custom parameters for this task."""
		return TaskParams()

	def generate_world(self, rng: jax.Array) -> EnvState:
		rng, _rng = jax.random.split(rng)
		builder = WorldBuilder(_rng, self.static_params, self.params)
		return builder.build(rng)
		# state = generate_world(rng, self.params, self.static_params)

		# return EnvState(
		# 	task_id=0,
		# 	map=state.map,
		# 	item_map=state.item_map,
		# 	mob_map=state.mob_map,
		# 	light_map=state.light_map,
		# 	down_ladders=state.down_ladders,
		# 	up_ladders=state.up_ladders,
		# 	chests_opened=state.chests_opened,
		# 	monsters_killed=state.monsters_killed,
		# 	player_position=state.player_position,
		# 	player_direction=state.player_direction,
		# 	player_level=state.player_level,
		# 	player_health=state.player_health,
		# 	player_food=state.player_food,
		# 	player_drink=state.player_drink,
		# 	player_energy=state.player_energy,
		# 	player_mana=state.player_mana,
		# 	player_recover=state.player_recover,
		# 	player_hunger=state.player_hunger,
		# 	player_thirst=state.player_thirst,
		# 	player_fatigue=state.player_fatigue,
		# 	player_recover_mana=state.player_recover_mana,
		# 	is_sleeping=state.is_sleeping,
		# 	is_resting=state.is_resting,
		# 	player_xp=state.player_xp,
		# 	player_dexterity=state.player_dexterity,
		# 	player_strength=state.player_strength,
		# 	player_intelligence=state.player_intelligence,
		# 	inventory=state.inventory,
		# 	sword_enchantment=state.sword_enchantment,
		# 	bow_enchantment=state.bow_enchantment,
		# 	armour_enchantments=state.armour_enchantments,
		# 	melee_mobs=state.melee_mobs,
		# 	ranged_mobs=state.ranged_mobs,
		# 	passive_mobs=state.passive_mobs,
		# 	mob_projectiles=state.mob_projectiles,
		# 	mob_projectile_directions=state.mob_projectile_directions,
		# 	player_projectiles=state.player_projectiles,
		# 	player_projectile_directions=state.player_projectile_directions,
		# 	growing_plants_positions=state.growing_plants_positions,
		# 	growing_plants_age=state.growing_plants_age,
		# 	growing_plants_mask=state.growing_plants_mask,
		# 	potion_mapping=state.potion_mapping,
		# 	learned_spells=state.learned_spells,
		# 	boss_progress=state.boss_progress,
		# 	boss_timesteps_to_spawn_this_round=state.boss_timesteps_to_spawn_this_round,
		# 	achievements=state.achievements,
		# 	light_level=state.light_level,
		# 	state_rng=state.state_rng,
		# 	timestep=state.timestep,
		# )
