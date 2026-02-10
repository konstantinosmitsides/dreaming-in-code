context = """
# CRAFTAX GAME DEFINITIONS

## TABLE 1: ACHIEVEMENTS
| Name | Category | Reward |
| :--- | :--- | :--- |
| COLLECT_WOOD | Basic | +1 |
| PLACE_TABLE | Basic | +1 |
| EAT_COW | Basic | +1 |
| COLLECT_SAPLING | Basic | +1 |
| COLLECT_DRINK | Basic | +1 |
| MAKE_WOOD_PICKAXE | Basic | +1 |
| MAKE_WOOD_SWORD | Basic | +1 |
| PLACE_PLANT | Basic | +1 |
| DEFEAT_ZOMBIE | Basic | +1 |
| COLLECT_STONE | Basic | +1 |
| PLACE_STONE | Basic | +1 |
| EAT_PLANT | Basic | +1 |
| DEFEAT_SKELETON | Basic | +1 |
| MAKE_STONE_PICKAXE | Basic | +1 |
| MAKE_STONE_SWORD | Basic | +1 |
| WAKE_UP | Basic | +1 |
| PLACE_FURNACE | Basic | +1 |
| COLLECT_COAL | Basic | +1 |
| COLLECT_IRON | Basic | +1 |
| COLLECT_DIAMOND | Basic | +1 |
| MAKE_IRON_PICKAXE | Basic | +1 |
| MAKE_IRON_SWORD | Basic | +1 |
| MAKE_ARROW | Basic | +1 |
| MAKE_TORCH | Basic | +1 |
| PLACE_TORCH | Basic | +1 |
| MAKE_DIAMOND_SWORD | Intermediate | +3 |
| MAKE_IRON_ARMOUR | Intermediate | +3 |
| MAKE_DIAMOND_ARMOUR | Intermediate | +3 |
| ENTER_GNOMISH_MINES | Intermediate | +3 |
| ENTER_DUNGEON | Intermediate | +3 |
| ENTER_SEWERS | Advanced | +5 |
| ENTER_VAULT | Advanced | +5 |
| ENTER_TROLL_MINES | Advanced | +5 |
| ENTER_FIRE_REALM | Very Advanced | +8 |
| ENTER_ICE_REALM | Very Advanced | +8 |
| ENTER_GRAVEYARD | Very Advanced | +8 |
| DEFEAT_GNOME_WARRIOR | Intermediate | +3 |
| DEFEAT_GNOME_ARCHER | Intermediate | +3 |
| DEFEAT_ORC_SOLIDER | Intermediate | +3 |
| DEFEAT_ORC_MAGE | Intermediate | +3 |
| DEFEAT_LIZARD | Advanced | +5 |
| DEFEAT_KOBOLD | Advanced | +5 |
| DEFEAT_TROLL | Advanced | +5 |
| DEFEAT_DEEP_THING | Advanced | +5 |
| DEFEAT_PIGMAN | Very Advanced | +8 |
| DEFEAT_FIRE_ELEMENTAL | Very Advanced | +8 |
| DEFEAT_FROST_TROLL | Very Advanced | +8 |
| DEFEAT_ICE_ELEMENTAL | Very Advanced | +8 |
| DAMAGE_NECROMANCER | Very Advanced | +8 |
| DEFEAT_NECROMANCER | Very Advanced | +8 |
| EAT_BAT | Intermediate | +3 |
| EAT_SNAIL | Intermediate | +3 |
| FIND_BOW | Intermediate | +3 |
| FIRE_BOW | Intermediate | +3 |
| COLLECT_SAPPHIRE | Intermediate | +3 |
| LEARN_FIREBALL | Advanced | +5 |
| CAST_FIREBALL | Advanced | +5 |
| LEARN_ICEBALL | Advanced | +5 |
| CAST_ICEBALL | Advanced | +5 |
| COLLECT_RUBY | Intermediate | +3 |
| MAKE_DIAMOND_PICKAXE | Intermediate | +3 |
| OPEN_CHEST | Intermediate | +3 |
| DRINK_POTION | Intermediate | +3 |
| ENCHANT_SWORD | Advanced | +5 |
| ENCHANT_ARMOUR | Advanced | +5 |
| DEFEAT_KNIGHT | Advanced | +5 |
| DEFEAT_ARCHER | Advanced | +5 |

## TABLE 2: BLOCKS
| Name | Cannot walk trough |
| :--- | :--- |
| INVALID | False |
| OUT_OF_BOUNDS | False |
| GRASS | False |
| WATER | False |
| STONE | True |
| TREE | True |
| WOOD | False |
| PATH | False |
| COAL | True |
| IRON | True |
| DIAMOND | True |
| CRAFTING_TABLE | True |
| FURNACE | True |
| SAND | False |
| LAVA | False |
| PLANT | True |
| RIPE_PLANT | True |
| WALL | True |
| DARKNESS | False |
| WALL_MOSS | True |
| STALAGMITE | True |
| SAPPHIRE | True |
| RUBY | True |
| CHEST | True |
| FOUNTAIN | True |
| FIRE_GRASS | False |
| ICE_GRASS | False |
| GRAVEL | False |
| FIRE_TREE | True |
| ICE_SHRUB | False |
| ENCHANTMENT_TABLE_FIRE | True |
| ENCHANTMENT_TABLE_ICE | True |
| NECROMANCER | True |
| GRAVE | True |
| GRAVE2 | True |
| GRAVE3 | True |
| NECROMANCER_VULNERABLE | False |

## TABLE 3: ACTIONS
| Name |
| :--- |
| NOOP |
| LEFT |
| RIGHT |
| UP |
| DOWN |
| DO |
| SLEEP |
| PLACE_STONE |
| PLACE_TABLE |
| PLACE_FURNACE |
| PLACE_PLANT |
| MAKE_WOOD_PICKAXE |
| MAKE_STONE_PICKAXE |
| MAKE_IRON_PICKAXE |
| MAKE_WOOD_SWORD |
| MAKE_STONE_SWORD |
| MAKE_IRON_SWORD |
| REST |
| DESCEND |
| ASCEND |
| MAKE_DIAMOND_PICKAXE |
| MAKE_DIAMOND_SWORD |
| MAKE_IRON_ARMOUR |
| MAKE_DIAMOND_ARMOUR |
| SHOOT_ARROW |
| MAKE_ARROW |
| CAST_FIREBALL |
| CAST_ICEBALL |
| PLACE_TORCH |
| DRINK_POTION_RED |
| DRINK_POTION_GREEN |
| DRINK_POTION_BLUE |
| DRINK_POTION_PINK |
| DRINK_POTION_CYAN |
| DRINK_POTION_YELLOW |
| READ_BOOK |
| ENCHANT_SWORD |
| ENCHANT_ARMOUR |
| MAKE_TORCH |
| LEVEL_UP_DEXTERITY |
| LEVEL_UP_STRENGTH |
| LEVEL_UP_INTELLIGENCE |
| ENCHANT_BOW |
""".strip()
