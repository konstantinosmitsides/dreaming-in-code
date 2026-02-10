context = """
================================================
FILE: craftax/craftax/constants.py
================================================
import os
import pathlib
from enum import Enum
import jax.numpy as jnp
import imageio.v3 as iio
import numpy as np
from PIL import Image
from craftax.craftax.util.maths_utils import get_distance_map
from craftax.environment_base.util import load_compressed_pickle, save_compressed_pickle

# GAME CONSTANTS
OBS_DIM = (9, 11)
assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1
MAX_OBS_DIM = max(OBS_DIM)
BLOCK_PIXEL_SIZE_HUMAN = 64
BLOCK_PIXEL_SIZE_IMG = 16
BLOCK_PIXEL_SIZE_AGENT = 10
INVENTORY_OBS_HEIGHT = 4
TEXTURE_CACHE_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "assets", "texture_cache.pbz2"
)

# ENUMS
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36

class ItemType(Enum):
    NONE = 0
    TORCH = 1
    LADDER_DOWN = 2
    LADDER_UP = 3
    LADDER_DOWN_BLOCKED = 4

class Action(Enum):
    NOOP = 0  #
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 8  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 5
    MAKE_STONE_SWORD = 15  # 6
    MAKE_IRON_SWORD = 16  # 7
    REST = 17  # e
    DESCEND = 18  # >
    ASCEND = 19  # <
    MAKE_DIAMOND_PICKAXE = 20  # 4
    MAKE_DIAMOND_SWORD = 21  # 8
    MAKE_IRON_ARMOUR = 22  # y
    MAKE_DIAMOND_ARMOUR = 23  # u
    SHOOT_ARROW = 24  # i
    MAKE_ARROW = 25  # o
    CAST_FIREBALL = 26  # g
    CAST_ICEBALL = 27  # h
    PLACE_TORCH = 28  # j
    DRINK_POTION_RED = 29  # z
    DRINK_POTION_GREEN = 30  # x
    DRINK_POTION_BLUE = 31  # c
    DRINK_POTION_PINK = 32  # v
    DRINK_POTION_CYAN = 33  # b
    DRINK_POTION_YELLOW = 34  # n
    READ_BOOK = 35  # m
    ENCHANT_SWORD = 36  # k
    ENCHANT_ARMOUR = 37  # l
    MAKE_TORCH = 38  # [
    LEVEL_UP_DEXTERITY = 39  # ]
    LEVEL_UP_STRENGTH = 40  # -
    LEVEL_UP_INTELLIGENCE = 41  # =
    ENCHANT_BOW = 42  # ;

class ProjectileType(Enum):
    ARROW = 0
    DAGGER = 1
    FIREBALL = 2
    ICEBALL = 3
    ARROW2 = 4
    SLIMEBALL = 5
    FIREBALL2 = 6
    ICEBALL2 = 7

# FLOOR MECHANICS

FLOOR_MOB_MAPPING = jnp.array(
    [
        # (passive, melee, ranged)
        jnp.array([0, 0, 0]),  # Floor 0 (overworld)
        jnp.array([2, 2, 2]),  # Floor 1 (dungeon)
        jnp.array([1, 1, 1]),  # Floor 2 (gnomish mines)
        jnp.array([2, 3, 3]),  # Floor 3 (sewers)
        jnp.array([2, 4, 4]),  # Floor 4 (vaults)
        jnp.array([1, 5, 5]),  # Floor 5 (troll mines)
        jnp.array([1, 6, 6]),  # Floor 6 (fire)
        jnp.array([1, 7, 7]),  # Floor 7 (ice)
        jnp.array([0, 0, 0]),  # Floor 8 (boss)
    ],
    dtype=jnp.int32,
)

FLOOR_MOB_SPAWN_CHANCE = jnp.array(
    [
        # (passive, melee, ranged, melee-night)
        jnp.array([0.1, 0.02, 0.05, 0.1]),  # Floor 0 (overworld)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 1 (gnomish mines)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 2 (dungeon)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 3 (sewers)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 4 (vaults)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 5 (troll mines)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 6 (fire)
        jnp.array([0.0, 0.06, 0.05, 0.0]),  # Floor 7 (ice)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

# Path blocks, water, lava  (everything collides with solid blocks)
COLLISION_LAND_CREATURE = [False, True, True]
COLLISION_FLYING = [False, False, False]
COLLISION_AQUATIC = [True, False, True]
COLLISION_AMPHIBIAN = [False, False, True]

MOB_TYPE_COLLISION_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, projectile)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 0 (overworld)
        jnp.array(
            [
                COLLISION_FLYING,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 1 (gnomish mines)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 2 (dungeon)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_AMPHIBIAN,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 3 (sewers)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 4 (vaults)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_AQUATIC,
                COLLISION_FLYING,
            ]
        ),  # Floor 5 (troll mines)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
                COLLISION_FLYING,
            ]
        ),  # Floor 6 (fire)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
                COLLISION_FLYING,
            ]
        ),  # Floor 7 (ice)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 8 (boss)
    ],
    dtype=jnp.int32,
)

NO_DAMAGE = jnp.array([0, 0, 0])
MOB_TYPE_DAMAGE_MAPPING = jnp.array(
    [
        # (-, melee, -, projectile)
        [NO_DAMAGE, [2, 0, 0], NO_DAMAGE, [2, 0, 0]],  # zombie, arrow
        [NO_DAMAGE, [4, 0, 0], NO_DAMAGE, [4, 0, 0]],  # gnome, dagger
        [NO_DAMAGE, [3, 0, 0], NO_DAMAGE, [0, 3, 0]],  # orc, fireball
        [NO_DAMAGE, [5, 0, 0], NO_DAMAGE, [0, 0, 3]],  # lizard, iceball
        [NO_DAMAGE, [6, 0, 0], NO_DAMAGE, [5, 0, 0]],  # knight, arrow2
        [NO_DAMAGE, [6, 1, 1], NO_DAMAGE, [4, 3, 3]],  # troll, slimeball
        [NO_DAMAGE, [3, 5, 0], NO_DAMAGE, [3, 5, 0]],  # pigman, fireball2
        [NO_DAMAGE, [4, 0, 5], NO_DAMAGE, [4, 0, 5]],  # ice troll, iceball2
    ],
    dtype=jnp.float32,
)

MOB_TYPE_HEALTH_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, -)
        jnp.array([3, 5, 3, 0]),  # Floor 0 (overworld)
        jnp.array([4, 7, 5, 0]),  # Floor 1 (gnomish mines)
        jnp.array([6, 9, 6, 0]),  # Floor 2 (dungeon)
        jnp.array([8, 11, 8, 0]),  # Floor 3 (sewers)
        jnp.array([0, 12, 12, 0]),  # Floor 4 (vaults)
        jnp.array([0, 20, 4, 0]),  # Floor 5 (troll mines)
        jnp.array([0, 20, 14, 0]),  # Floor 6 (fire)
        jnp.array([0, 24, 16, 0]),  # Floor 7 (ice)
        jnp.array([0, 0, 0, 0]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

NO_DEFENSE = [0, 0, 0]
MOB_TYPE_DEFENSE_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, -)
        jnp.array(
            [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]
        ),  # Floor 0 (overworld)
        jnp.array(
            [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]
        ),  # Floor 1 (gnomish mines)
        jnp.array(
            [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]
        ),  # Floor 2 (dungeon)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 3 (sewers)
        jnp.array(
            [NO_DEFENSE, [0.5, 0, 0], [0.5, 0, 0], NO_DEFENSE]
        ),  # Floor 4 (vaults)
        jnp.array(
            [NO_DEFENSE, [0.2, 0, 0], [0.0, 0.0, 0.0], NO_DEFENSE]
        ),  # Floor 5 (troll mines)
        jnp.array(
            [NO_DEFENSE, [0.9, 1.0, 0.0], [0.9, 1.0, 0.0], NO_DEFENSE]
        ),  # Floor 6 (fire)
        jnp.array(
            [NO_DEFENSE, [0.9, 0.0, 1.0], [0.9, 0.0, 1.0], NO_DEFENSE]
        ),  # Floor 7 (ice)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING = jnp.array(
    [
        0,  # Skeleton --> Arrow
        0,  # Gnome archer --> Arrow
        2,  # Orc mage --> Fireball
        1,  # Kobold --> Dagger
        4,  # Knight archer --> Arrow2
        5,  # Deep thing --> Slime ball
        6,  # Fire elemental --> Fireball2
        7,  # Ice elemental --> Iceball2
    ]
)

# GAME MECHANICS
BOSS_FIGHT_EXTRA_DAMAGE = 0.5
BOSS_FIGHT_SPAWN_TURNS = 7

DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
        jnp.zeros((11, 2), dtype=jnp.int32),
    ),
    axis=0,
)

CLOSE_BLOCKS = jnp.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=jnp.int32,
)

# Can't walk through these
SOLID_BLOCKS = [
    BlockType.STONE.value,
    BlockType.TREE.value,
    BlockType.COAL.value,
    BlockType.IRON.value,
    BlockType.DIAMOND.value,
    BlockType.CRAFTING_TABLE.value,
    BlockType.FURNACE.value,
    BlockType.PLANT.value,
    BlockType.RIPE_PLANT.value,
    BlockType.WALL.value,
    BlockType.WALL_MOSS.value,
    BlockType.STALAGMITE.value,
    BlockType.RUBY.value,
    BlockType.SAPPHIRE.value,
    BlockType.CHEST.value,
    BlockType.FOUNTAIN.value,
    BlockType.FIRE_TREE.value,
    BlockType.ENCHANTMENT_TABLE_FIRE.value,
    BlockType.ENCHANTMENT_TABLE_ICE.value,
    BlockType.GRAVE.value,
    BlockType.GRAVE2.value,
    BlockType.GRAVE3.value,
    BlockType.NECROMANCER.value,
]

SOLID_BLOCK_MAPPING = jnp.array(
    [(block.value in SOLID_BLOCKS) for block in BlockType], dtype=bool
)

CAN_PLACE_ITEM_BLOCKS = [
    BlockType.GRASS.value,
    BlockType.SAND.value,
    BlockType.PATH.value,
    BlockType.FIRE_GRASS.value,
    BlockType.ICE_GRASS.value,
]

CAN_PLACE_ITEM_MAPPING = jnp.array(
    [(block.value in CAN_PLACE_ITEM_BLOCKS) for block in BlockType], dtype=bool
)

# ACHIEVEMENTS
class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21

    MAKE_ARROW = 22
    MAKE_TORCH = 23
    PLACE_TORCH = 24

    COLLECT_SAPPHIRE = 54
    COLLECT_RUBY = 59
    MAKE_DIAMOND_PICKAXE = 60
    MAKE_DIAMOND_SWORD = 25
    MAKE_IRON_ARMOUR = 26
    MAKE_DIAMOND_ARMOUR = 27

    ENTER_GNOMISH_MINES = 28
    ENTER_DUNGEON = 29
    ENTER_SEWERS = 30
    ENTER_VAULT = 31
    ENTER_TROLL_MINES = 32
    ENTER_FIRE_REALM = 33
    ENTER_ICE_REALM = 34
    ENTER_GRAVEYARD = 35

    DEFEAT_GNOME_WARRIOR = 36
    DEFEAT_GNOME_ARCHER = 37
    DEFEAT_ORC_SOLIDER = 38
    DEFEAT_ORC_MAGE = 39
    DEFEAT_LIZARD = 40
    DEFEAT_KOBOLD = 41
    DEFEAT_KNIGHT = 65
    DEFEAT_ARCHER = 66
    DEFEAT_TROLL = 42
    DEFEAT_DEEP_THING = 43
    DEFEAT_PIGMAN = 44
    DEFEAT_FIRE_ELEMENTAL = 45
    DEFEAT_FROST_TROLL = 46
    DEFEAT_ICE_ELEMENTAL = 47
    DAMAGE_NECROMANCER = 48
    DEFEAT_NECROMANCER = 49

    EAT_BAT = 50
    EAT_SNAIL = 51

    FIND_BOW = 52
    FIRE_BOW = 53

    LEARN_FIREBALL = 55
    CAST_FIREBALL = 56
    LEARN_ICEBALL = 57
    CAST_ICEBALL = 58

    OPEN_CHEST = 61
    DRINK_POTION = 62
    ENCHANT_SWORD = 63
    ENCHANT_ARMOUR = 64

INTERMEDIATE_ACHIEVEMENTS = [
    Achievement.COLLECT_SAPPHIRE.value,
    Achievement.COLLECT_RUBY.value,
    Achievement.MAKE_DIAMOND_PICKAXE.value,
    Achievement.MAKE_DIAMOND_SWORD.value,
    Achievement.MAKE_IRON_ARMOUR.value,
    Achievement.MAKE_DIAMOND_ARMOUR.value,
    Achievement.ENTER_GNOMISH_MINES.value,
    Achievement.ENTER_DUNGEON.value,
    Achievement.DEFEAT_GNOME_WARRIOR.value,
    Achievement.DEFEAT_GNOME_ARCHER.value,
    Achievement.DEFEAT_ORC_SOLIDER.value,
    Achievement.DEFEAT_ORC_MAGE.value,
    Achievement.EAT_BAT.value,
    Achievement.EAT_SNAIL.value,
    Achievement.FIND_BOW.value,
    Achievement.FIRE_BOW.value,
    Achievement.OPEN_CHEST.value,
    Achievement.DRINK_POTION.value,
]

VERY_ADVANCED_ACHIEVEMENTS = [
    Achievement.ENTER_FIRE_REALM.value,
    Achievement.ENTER_ICE_REALM.value,
    Achievement.ENTER_GRAVEYARD.value,
    Achievement.DEFEAT_PIGMAN.value,
    Achievement.DEFEAT_FIRE_ELEMENTAL.value,
    Achievement.DEFEAT_FROST_TROLL.value,
    Achievement.DEFEAT_ICE_ELEMENTAL.value,
    Achievement.DAMAGE_NECROMANCER.value,
    Achievement.DEFEAT_NECROMANCER.value,
]

def achievement_mapping(achievement_value):
    if achievement_value <= 24:
        return 1
    elif achievement_value in INTERMEDIATE_ACHIEVEMENTS:
        return 3
    elif achievement_value in VERY_ADVANCED_ACHIEVEMENTS:
        return 8
    else:
        return 5

ACHIEVEMENT_REWARD_MAP = jnp.array(
    [achievement_mapping(i) for i in range(len(Achievement))]
)

LEVEL_ACHIEVEMENT_MAP = jnp.array(
    [
        0,
        Achievement.ENTER_DUNGEON.value,
        Achievement.ENTER_GNOMISH_MINES.value,
        Achievement.ENTER_SEWERS.value,
        Achievement.ENTER_VAULT.value,
        Achievement.ENTER_TROLL_MINES.value,
        Achievement.ENTER_FIRE_REALM.value,
        Achievement.ENTER_ICE_REALM.value,
        Achievement.ENTER_GRAVEYARD.value,
    ]
)

MOB_ACHIEVEMENT_MAP = jnp.array(
    [
        # Passive
        [
            Achievement.EAT_COW.value,
            Achievement.EAT_BAT.value,
            Achievement.EAT_SNAIL.value,
            0,
            0,
            0,
            0,
            0,
        ],
        # Melee
        [
            Achievement.DEFEAT_ZOMBIE.value,
            Achievement.DEFEAT_GNOME_WARRIOR.value,
            Achievement.DEFEAT_ORC_SOLIDER.value,
            Achievement.DEFEAT_LIZARD.value,
            Achievement.DEFEAT_KNIGHT.value,
            Achievement.DEFEAT_TROLL.value,
            Achievement.DEFEAT_PIGMAN.value,
            Achievement.DEFEAT_FROST_TROLL.value,
        ],
        # Ranged
        [
            Achievement.DEFEAT_SKELETON.value,
            Achievement.DEFEAT_GNOME_ARCHER.value,
            Achievement.DEFEAT_ORC_MAGE.value,
            Achievement.DEFEAT_KOBOLD.value,
            Achievement.DEFEAT_ARCHER.value,
            Achievement.DEFEAT_DEEP_THING.value,
            Achievement.DEFEAT_FIRE_ELEMENTAL.value,
            Achievement.DEFEAT_ICE_ELEMENTAL.value,
        ],
    ]
)

# PRE-COMPUTATION
TORCH_LIGHT_MAP = get_distance_map(jnp.array([4, 4]), (9, 9))
TORCH_LIGHT_MAP /= 5.0
TORCH_LIGHT_MAP = jnp.clip(1 - TORCH_LIGHT_MAP, 0.0, 1.0)

================================================
FILE: craftax/craftax/craftax_state.py
================================================
from dataclasses import dataclass
from typing import Tuple, Any

import jax
from flax import struct
import jax.numpy as jnp

@struct.dataclass
class Inventory:
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    pickaxe: int
    sword: int
    bow: int
    arrows: int
    armour: jnp.ndarray
    torches: int
    ruby: int
    sapphire: int
    potions: jnp.ndarray
    books: int

@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: jnp.ndarray
    mask: jnp.ndarray
    attack_cooldown: jnp.ndarray
    type_id: jnp.ndarray

@struct.dataclass
class EnvState:
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

@struct.dataclass
class EnvParams:
    max_timesteps: int = 100000
    day_length: int = 300

    always_diamond: bool = False

    mob_despawn_distance: int = 14
    max_attribute: int = 5

    god_mode: bool = False

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)

@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (48, 48)
    num_levels: int = 9

    # Mobs
    max_melee_mobs: int = 3
    max_passive_mobs: int = 3
    max_growing_plants: int = 10
    max_ranged_mobs: int = 2
    max_mob_projectiles: int = 3
    max_player_projectiles: int = 3
"""
