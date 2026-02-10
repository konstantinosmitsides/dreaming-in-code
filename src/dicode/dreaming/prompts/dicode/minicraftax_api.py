context = """
To generate valid and interesting tasks, you must first understand the capabilities of this API.

These tasks are defined by two components:
1.  **`TaskParams`**: A set of parameters that modify the core game mechanics (e.g., making mobs harder, making survival needs more pressing).
2.  **`WorldBuilder`**: A class used to programmatically set up the initial state of the world (e.g., placing specific blocks, or pre-filling the player's inventory).

## MiniCraftax API Documentation

Below is the documentation for the `TaskParams` and `WorldBuilder` classes. Review it carefully.

---

### `TaskParams`: Modifying Game Mechanics

The `TaskParams` class allows you to tweak the game's dynamic rules. You can think of this as setting the "difficulty" or "ruleset" for a specific task. The default values are the original Craftax game.

* `passive_spawn_multiplier: float`
    * **Effect:** Scales the base spawn chance for **passive mobs** (like cows). A value of `2.0` doubles the spawn rate, while `0.5` halves it. `0` means no passive mobs and any existing passive mobs are removed if agent goes far from them.
    * **Default:** `1.0`

* `melee_spawn_multiplier: float`
    * **Effect:** Scales the base spawn chance for **melee mobs** (like zombies). `0` means no melee mobs and any existing melee mobs are removed if agent goes far from them.
    * **Default:** `1.0`

* `ranged_spawn_multiplier: float`
    * **Effect:** Scales the base spawn chance for **ranged mobs** (like skeletons). `0` means no ranged mobs and any existing ranged mobs are removed if agent goes far from them.
    * **Default:** `1.0`

* `mob_health_multiplier: float`
    * **Effect:** Multiplies the base health of all mobs (passive, melee, and ranged) when they spawn. A value of `2.0` means mobs spawn with double health.
    * **Default:** `1.0`

* `mob_damage_multiplier: float`
    * **Effect:** Multiplies the base damage of all mob attacks (both melee and ranged projectiles). A value of `2.0` doubles mob damage.
    * **Default:** `1.0`

* `melee_trigger_distance: int`
    * **Effect:** The Manhattan distance at which melee mobs will detect the player and begin chasing them.
    * **Default:** `10`

* `monsters_killed_to_clear_level: int`
    * **Effect:** The number of hostile monsters (melee or ranged) the player must defeat on a level to unlock the "ladder down" to the next floor. 
    * **Default:** `8` except for floor 0 which is `0`.

* `needs_depletion_multiplier: float`
    * **Effect:** Scales the rate at which the player's **hunger, thirst, and fatigue** increase. A value of `2.0` means the player gets hungry, thirsty, and tired twice as fast.
    * **Default:** `1.0`

* `health_recover_multiplier: float`
    * **Effect:** Scales the speed of the player's natural **health regeneration** (which only occurs when all needs are met).
    * **Default:** `1.0`

* `health_loss_multiplier: float`
    * **Effect:** Scales the speed at which the player **loses health** from unmet needs (e.g., starvation or dehydration).
    * **Default:** `1.0`

* `mana_recover_multiplier: float`
    * **Effect:** Scales the speed of the player's natural **mana regeneration**.
    * **Default:** `1.0`

* `growing_plants_age: int`
    * **Effect:** The number of game steps (ticks) it takes for a planted `PLANT` to mature into a `RIPE_PLANT`.
    * **Default:** `600`

---

### `WorldBuilder`: Setting the Initial State

The `WorldBuilder` class provides methods to modify the initial state of the world.

* `set_starting_floor(self, level: int)`
    * **Effect:** Sets the player's starting floor. If `level > 0`, the player will spawn at that level's "up ladder" position.
    * **Example:** `builder.set_starting_floor(3)` starts the player in the Sewers.

* `set_player_stats(self, dexterity: int = 1, strength: int = 1, intelligence: int = 1)`
    * **Effect:** Sets the player's starting attributes (DEX, STR, INT). Values are clamped between 1 and 5.
    * **Example:** `builder.set_player_stats(strength=3)` gives the player a starting strength boost.

* `set_player_inventory(self, inventory_dict: dict)`
    * **Effect:** Sets the player's starting inventory. Takes a dictionary where keys are item names (e.g., "wood", "stone", "pickaxe") and values are the counts.
    * **Example:** `builder.set_player_inventory({"stone": 20, "pickaxe": 1})`

* `set_weapon_enchantments(self, sword: int = 0, bow: int = 0)`
    * **Effect:** Sets the player's starting weapon enchantments. (0 = None, 1 = Fire, 2 = Ice).
    * **Example:** `builder.set_weapon_enchantments(sword=1)` starts the player with a fire-enchanted sword.

* `set_armour_enchantments(self, helmet: int = 0, chestplate: int = 0, leggings: int = 0, boots: int = 0)`
    * **Effect:** Sets enchantments for each armour slot. (0 = None, 1 = Fire, 2 = Ice).

* `set_learned_spells(self, fireball: bool = False, iceball: bool = False)`
    * **Effect:** Sets whether the player starts the game having already learned the Fireball or Iceball spells.
    * **Example:** `builder.set_learned_spells(fireball=True)`

* `set_monsters_killed(self, level: int, count: int)`
    * **Effect:** Manually sets the monster kill count for a specific `level`. This can be used to pre-unlock the "ladder down" for that level.
    * **Example:** `builder.set_monsters_killed(level=0, count=8)` unlocks the ladder from the Overworld immediately.

* `place_block(self, level: int, block_type: BlockType, position: tuple)`
    * **Effect:** Places a *single* block (e.g., `BlockType.CRAFTING_TABLE`) at an exact (row, col) `position` on a specific `level`.
    * **Example:** `builder.place_block(0, BlockType.DIAMOND, (24, 25))`

* `fill_area(self, level: int, block_type: BlockType, top_left: tuple, bottom_right: tuple)`
    * **Effect:** Fills a rectangular area on a specific `level` with a `block_type`.
    * **Example:** `builder.fill_area(0, BlockType.WATER, (10, 10), (20, 20))` creates a lake.

* `place_randomly(self, rng: jax.Array, level: int, block_type: BlockType, n: int = 1, on_blocks: List[BlockType] = ...)`
    * **Effect:** Places `n` blocks of `block_type` at random locations on the specified `level`. The blocks are only placed on top of blocks specified in the `on_blocks` list (e.g., `[BlockType.GRASS]`).
    * **Example:** `builder.place_randomly(rng, 0, BlockType.TREE, 50, [BlockType.GRASS])`

* `place_randomly_near(self, rng: jax.Array, level: int, block_type: BlockType, target_pos: tuple, min_dist: int, max_dist: int, n: int = 1, on_blocks: List[BlockType] = ...)`
    * **Effect:** Places `n` blocks randomly within a Manhattan distance range (`min_dist` to `max_dist`) from a `target_pos` (row, col).
    * **Example:** `builder.place_randomly_near(rng, 0, BlockType.COAL, (24, 24), 2, 5, 10, [BlockType.STONE])`

* `add_mobs_randomly_near(self, rng: jax.Array, level: int, mob_name: str, type_id: int, n: int = 1, target_pos: jnp.ndarray = None, min_dist: int = 0, max_dist: int = 5, on_blocks: List[BlockType] = ...)`
    * **Effect:** Adds `n` mobs of `mob_name` randomly within a Manhattan distance range (`min_dist` to `max_dist`) from `target_pos`. If `target_pos` is `None`, defaults to player position.
    * **Example:** `builder.add_mobs_randomly_near(rng, 0, "melee", MobType.MELEE.value, 5, (30, 30), 2, 8, [BlockType.GRASS])`
    * **NOTES:** 
        * we can place maximum 3 melee mobs, 3 passive mobs, and 2 ranged mobs
        * once the mobs are placed, the spawning and update logic of mobs works as normal and thus the inital mobs might be removed if the player is not close enough to them

* `place_adjacent_to_existing(self, rng: jax.Array, level: int, block_to_place: BlockType, target_block_type: BlockType, on_blocks: List[BlockType] = ...)`
    * **Effect:** Finds one existing `target_block_type` and places a `block_to_place` in a valid, random adjacent spot.
    * **Example:** `builder.place_adjacent_to_existing(rng, 0, BlockType.FURNACE, BlockType.CRAFTING_TABLE, [BlockType.GRASS])`

* `build(self, rng: jax.Array)`
    * **Effect:** Finalizes the world and returns the complete `EnvState` object. This is the final call in any world-building chain.
"""
