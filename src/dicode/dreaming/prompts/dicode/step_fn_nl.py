context = """
The following descriptions define the core code structure of the `env.step` function.

**How to interpret this:**
1.  **The Logic Flow is Fixed:** You cannot change *how* the game processes actions (e.g., you cannot rewrite the code for `update_mobs` to make zombies fly).
2.  **The Parameters are Mutable:** However, this logic relies on variables from `TaskParams` (e.g., `update_mobs` uses `task.mob_damage_multiplier`). You **CAN** change those values via the API to control the difficulty of these mechanics.



The `change_floor` function manages vertical movement between dungeon levels based on the agent's `action`.

**Descending Logic (`Action.DESCEND`):**
The agent attempts to move to the next deeper level (`player_level + 1`).
- **Prerequisites:** 1. Current level is not the last level (`num_levels - 1`).
  2. Either `env_params.god_mode` is True OR:
     - The agent is standing on a `LADDER_DOWN` tile.
     - AND the agent has killed the required number of monsters (`task.monsters_killed_to_clear_level`) for the current level.
- **Result:**
  - Player level increases by 1.
  - Player coordinates are set to the location of the `LADDER_UP` on the new level.

**Ascending Logic (`Action.ASCEND`):**
The agent attempts to return to the previous level (`player_level - 1`).
- **Prerequisites:**
  1. Current level is greater than 0.
  2. Either `env_params.god_mode` is True OR the agent is standing on a `LADDER_UP` tile.
- **Result:**
  - Player level decreases by 1.
  - Player coordinates are set to the location of the `LADDER_DOWN` on the new level.

**State Updates & Rewards:**
- If neither movement condition is met, position and level remain unchanged.
- **Exploration Reward:** If the agent successfully descends to a level (other than level 0) that is visited for the first time (checked via `achievements`), `player_xp` is increased by +1 and the level is marked as achieved.


The `do_crafting` function handles the creation of tools, weapons, armor, and consumables. Success depends on the agent's **proximity to specific blocks**, possessing sufficient **resources**, and the specific **Action** triggered.

**General Crafting Rules:**
* **Station Proximity:** All recipes require standing near a **Crafting Table**.
    * **Exception:** **Iron** items (Pickaxe, Sword, Armour) require standing near **BOTH** a **Crafting Table** and a **Furnace**.
* **Upgrade Logic:** Tools and weapons are only crafted if the agent's current tier is lower than the tier being crafted (e.g., you cannot craft a Stone Pickaxe if you already have an Iron one).
* **Armor Logic:** Armor is crafted for the first available inventory slot that is below the target tier.

**Recipe Reference Table:**

| Item Category | Item Name | Action | Ingredients Consumed | Station | Yield / Effect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pickaxes** | Wood Pickaxe | `MAKE_WOOD_PICKAXE` | 1 Wood | Table | Level 1 Pickaxe |
| | Stone Pickaxe | `MAKE_STONE_PICKAXE` | 1 Wood, 1 Stone | Table | Level 2 Pickaxe |
| | Iron Pickaxe | `MAKE_IRON_PICKAXE` | 1 Wood, 1 Stone, 1 Iron, 1 Coal | **Table + Furnace** | Level 3 Pickaxe |
| | Diamond Pickaxe | `MAKE_DIAMOND_PICKAXE` | 1 Wood, 3 Diamond | Table | Level 4 Pickaxe |
| **Swords** | Wood Sword | `MAKE_WOOD_SWORD` | 1 Wood | Table | Level 1 Sword |
| | Stone Sword | `MAKE_STONE_SWORD` | 1 Wood, 1 Stone | Table | Level 2 Sword |
| | Iron Sword | `MAKE_IRON_SWORD` | 1 Wood, 1 Stone, 1 Iron, 1 Coal | **Table + Furnace** | Level 3 Sword |
| | Diamond Sword | `MAKE_DIAMOND_SWORD` | 1 Wood, 2 Diamond | Table | Level 4 Sword |
| **Armour** | Iron Armour | `MAKE_IRON_ARMOUR` | 3 Iron, 3 Coal | **Table + Furnace** | +1 Defence (fills slot) |
| | Diamond Armour | `MAKE_DIAMOND_ARMOUR` | 3 Diamond | Table | +2 Defence (fills slot) |
| **Consumables** | Arrows | `MAKE_ARROW` | 1 Wood, 1 Stone | Table | **+2** Arrows (Max 99) |
| | Torches | `MAKE_TORCH` | 1 Wood, 1 Coal | Table | **+4** Torches (Max 99) |

**Achievements:**
Crafting Iron Armour or Diamond Armour triggers their respective achievements (`MAKE_IRON_ARMOUR`, `MAKE_DIAMOND_ARMOUR`).

The `do_action` function executes the **`Action.DO`** command, allowing the agent to interact with the block directly in front of them (the "target block").

**Priority Logic:**
1.  **Combat First:** The system first attempts to attack a mob in the target block (via `attack_mob`).
2.  **Block Interaction:** If **no mob was attacked**, the agent interacts with the static block in the target location.

**Mining & Harvesting Rules:**
Mining permanently removes the block (replacing it with `PATH` or `GRASS`) and adds items to the inventory, provided the agent meets the **Tool Requirement**.

| Target Block | Tool Requirement | Inventory Gain | Replacement Block |
| :--- | :--- | :--- | :--- |
| **Tree** (Normal/Fire/Ice) | None | +1 Wood | Grass / Fire Grass / Ice Grass |
| **Stone** | Pickaxe Level $\\ge$ 1 | +1 Stone | Path |
| **Coal** | Pickaxe Level $\\ge$ 1 | +1 Coal | Path |
| **Stalagmite** | Pickaxe Level $\\ge$ 1 | +1 Stone | Path |
| **Iron** | Pickaxe Level $\\ge$ 2 | +1 Iron | Path |
| **Diamond** | Pickaxe Level $\\ge$ 3 | +1 Diamond | Path |
| **Sapphire / Ruby** | Pickaxe Level $\\ge$ 4 | +1 Sapphire / Ruby | Path |
| **Grass** | None | 10% Chance: +1 Sapling | (Remains Grass) |

**Consumables & Restoration:**
* **Water / Fountain:** Drinking fills `player_drink` to max and resets `player_thirst` to 0. (Achievement: `COLLECT_DRINK`)
* **Ripe Plant:** Eating adds +4 `player_food` (up to max), resets `player_hunger` to 0, and reverts the block to an unripe `PLANT`. (Achievement: `EAT_PLANT`)

**Other Interactions:**
* **Chest:** Opens the chest, removing it from the map and granting random loot via `add_items_from_chest`. (Achievement: `OPEN_CHEST`)
* **Workstations (Furnace / Crafting Table):** "Mining" these destroys them (replaces with `PATH`) without returning resources.
* **Boss (Necromancer):** If the boss is in the target block, is vulnerable, and the fight is active, the agent deals damage, incrementing `boss_progress`. (Achievement: `DAMAGE_NECROMANCER`)

The `place_block` function executes specific **Placement Actions** to modify the environment in the cell directly **in front of** the agent (the "target cell").

**General Validation:**
All placement actions fail (resulting in no state change) if:
1.  The target cell is out of bounds.
2.  There is a **Mob** in the target cell.
3.  (For most blocks) The target cell already contains a solid block or item.

**Placement Logic & Costs:**

| Action | Cost | Target Requirement | Result |
| :--- | :--- | :--- | :--- |
| **PLACE_TABLE** | 2 Wood | Empty, Valid Terrain | Target block becomes **Crafting Table**. |
| **PLACE_FURNACE** | 1 Stone | Empty, Valid Terrain | Target block becomes **Furnace**. |
| **PLACE_STONE** | 1 Stone | Empty **OR Water** | Target block becomes **Stone**. (Allows bridging over water). |
| **PLACE_TORCH** | 1 Torch | Valid Surface, No Item | Adds **Torch** to `item_map`. Updates **Light Map**. |
| **PLACE_PLANT** | 1 Sapling | **Grass** Block, No Item | Target block becomes **Plant**. Registers plant for growth. |

**Special Mechanics:**
* **Torches & Lighting:** Placing a torch instantly updates the `light_map` by adding a specific light gradient (`TORCH_LIGHT_MAP`) centered on the torch, clipped between 0.0 and 1.0. 
* **Plants:** Placing a sapling on grass initiates the farming cycle. The plant is added to `growing_plants_positions` and initialized with an age of 0.
* **Achievements:** Successfully placing any of these items triggers the corresponding Achievement.

The `shoot_projectile` function manages the mechanics of firing ranged weapons (Arrows).

**Trigger Condition:** Action is `SHOOT_ARROW`.

**Prerequisites (ALL must be met):**
1.  **Equipment:** The agent possesses a Bow (`inventory.bow >= 1`).
2.  **Ammo:** The agent possesses at least one Arrow (`inventory.arrows >= 1`).
3.  **Projectile Limit:** The number of active player projectiles on the current level is below the hard cap (`static_params.max_player_projectiles`).

**Execution Results:**
* **Spawn:** A new projectile (Type: `ARROW2`) is instantiated at the player's current coordinates.
* **Trajectory:** The projectile is assigned a velocity vector matching the agent's current facing direction (`player_direction`).
* **Cost:** Inventory `arrows` count is decremented by 1.
* **Achievement:** Unlocks the `FIRE_BOW` achievement.

**Failure Case:** If any prerequisite is not met, the action is ignored, and no ammunition is consumed.

The `cast_spell` function handles the deployment of magic projectiles (`FIREBALL` or `ICEBALL`) based on the agent's action.

**Prerequisites (ALL must be met):**
1.  **Mana:** The agent must have at least **2 Mana**.
2.  **Knowledge:** The specific spell must be unlocked in the agent's `learned_spells` array (Index 0 for Fireball, Index 1 for Iceball).
3.  **Projectile Limit:** The active projectile count must be below the hard cap (`static_params.max_player_projectiles`).

**Spell Mechanics:**

| Spell | Action Trigger | Cost | Projectile Type |
| :--- | :--- | :--- | :--- |
| **Fireball** | `CAST_FIREBALL` | 2 Mana | `FIREBALL` |
| **Iceball** | `CAST_ICEBALL` | 2 Mana | `ICEBALL` |

**Execution Results:**
* **Spawn:** Creates a projectile of the determined type at the player's current position.
* **Trajectory:** The projectile moves in the direction the player is currently facing.
* **Cost:** Deducts **2 Mana** from `player_mana`.
* **Achievement:** Unlocks `CAST_FIREBALL` or `CAST_ICEBALL` upon successful cast.

The `drink_potion` function handles the consumption of consumables, featuring a **Roguelike Potion Mechanics** system where the relationship between potion *Color* and potion *Effect* is randomized via `state.potion_mapping`.

**Trigger & Validation:**
* **Actions:** 6 distinct actions exist: `DRINK_POTION_[COLOR]` (Red, Green, Blue, Pink, Cyan, Yellow).
* **Prerequisite:** The agent must possess at least **1 unit** of the specific potion color in their inventory.

**Effect Resolution:**
The specific effect applied is determined by looking up the potion's color index in `state.potion_mapping`.

| Effect Index | Stat Affected | Change | Description |
| :--- | :--- | :--- | :--- |
| **0** | Health | **+8** | Major Heal |
| **1** | Health | **-3** | Poison / Damage |
| **2** | Mana | **+8** | Restore Mana |
| **3** | Mana | **-3** | Drain Mana |
| **4** | Energy | **+8** | Restore Energy |
| **5** | Energy | **-3** | Drain Energy |

**State Updates:**
* **Inventory:** Decrements the count of the specific potion color consumed.
* **Stats:** Updates `player_health`, `player_mana`, and `player_energy` based on the resolved effect.
* **Achievement:** Unlocks the `DRINK_POTION` achievement.

The `read_book` function allows the agent to permanently unlock new magic abilities by consuming a Book.

**Trigger & Validation:**
* **Action:** `READ_BOOK`
* **Prerequisite:** The agent possesses at least **1 Book** in their inventory.

**Learning Mechanic:**
The system randomly selects one spell from the set of spells the agent has **not yet learned**.
* **Index 0:** Fireball
* **Index 1:** Iceball

**Execution Results:**
* **Inventory:** Consumes 1 Book.
* **Knowledge:** The selected spell is set to `True` in `state.learned_spells`, enabling the corresponding `CAST_` action.
* **Achievements:** Unlocks either `LEARN_FIREBALL` or `LEARN_ICEBALL` based on the spell acquired.

The `enchant` function allows the agent to imbue weapons and armor with elemental properties (Fire or Ice) by interacting with a specific **Enchantment Table** block directly in front of them.

**Prerequisites (ALL must be met):**
1.  **Position:** Agent must be facing an `ENCHANTMENT_TABLE_FIRE` or `ENCHANTMENT_TABLE_ICE`.
2.  **Mana:** Agent must have at least **9 Mana**.
3.  **Gem:** Agent must possess the specific gem required by the table type.
    * **Fire Table:** Requires 1 **Ruby**.
    * **Ice Table:** Requires 1 **Sapphire**.
4.  **Item:** Agent must possess the item corresponding to the action (Sword, Bow, or at least one Armour piece).

**Enchantment Mechanics:**
* **Element Type:** Defined by the table (Fire Table $\to$ Type 1, Ice Table $\to$ Type 2).
* **Sword / Bow:** Applies the enchantment type to the item, overwriting any previous enchantment.
* **Armour:** The system selects a single armor slot to enchant based on the following priority:
    1.  **Unenchanted Slots:** Randomly selects a slot that currently has no enchantment.
    2.  **Overwrite:** If (and only if) all slots are full, it randomly selects a slot containing the **opposite** element to overwrite.

**Costs & Rewards:**
* **Resources:** Consumes **1 Gem** (Ruby/Sapphire) and **9 Mana**.
* **Achievements:** Triggers `ENCHANT_SWORD` or `ENCHANT_ARMOUR` upon success.

The `boss_logic` function manages the passive state updates regarding the **Necromancer Boss**.

**Logic:**
1.  **Fight Timer:** If the boss fight is currently active (`is_fighting_boss`), the internal timer `boss_timesteps_to_spawn_this_round` is decremented by 1.
2.  **Victory Check:** The system evaluates if the victory condition is met (via `has_beaten_boss`). If true, the `DEFEAT_NECROMANCER` achievement is unlocked.

The `level_up_attributes` function manages character progression, allowing the agent to convert Experience Points (XP) into permanent stat increases.

**Prerequisites:**
1.  **Cost:** The agent must have at least **1 XP**.
2.  **Cap:** The target attribute must be below the maximum limit defined by `params.max_attribute`.

**Actions & Effects:**

| Action | Attribute Increased | Effect Description (Implicit) |
| :--- | :--- | :--- |
| `LEVEL_UP_DEXTERITY` | **+1 Dexterity** | typically increases ranged damage/accuracy. |
| `LEVEL_UP_STRENGTH` | **+1 Strength** | typically increases melee damage. |
| `LEVEL_UP_INTELLIGENCE` | **+1 Intelligence** | typically increases magic potential/mana. |

**Execution:**
If the action is valid and prerequisites are met, the specific attribute is incremented by 1, and `player_xp` is decremented by 1.

The `move_player` function handles the agent's attempt to move horizontally within the current level using directional commands (Up, Down, Left, Right).

**Logic Flow:**
1.  **Proposed Movement:** Calculates the target coordinate based on the `action` vector.
2.  **Validation:** Checks if the target coordinate is valid via `is_position_in_bounds_not_in_mob_not_colliding`. A move is valid if:
    * It is within map boundaries.
    * It is **not** occupied by a Mob.
    * It is **not** a solid block (Collision logic: `COLLISION_LAND_CREATURE`).
    * *Override:* If `params.god_mode` is active, collisions are ignored.
3.  **Position Update:**
    * **Success:** If valid, the agent's coordinate is updated.
    * **Failure:** If invalid (blocked), the agent remains at the current coordinate.
4.  **Direction Update:**
    * The agent's `player_direction` is updated to match the input `action`, **regardless of whether the move succeeded**. This allows the agent to turn and face a wall without moving into it.

The `update_mobs` function executes the AI cycles for all entities in the environment. It processes groups sequentially: Melee Mobs, Passive Mobs, Ranged Mobs, Mob Projectiles, and Player Projectiles.

**1. Melee Mobs (Zombies, Skeletons, etc.)**
* **AI Logic:**
    * **Chase:** If within trigger distance (<`task.melee_trigger_distance` blocks) or fighting a boss, there is a **75% chance** to move directly toward the player.
    * **Wander:** Otherwise (or if the 75% check fails), moves in a random cardinal direction.
* **Attack:** If adjacent (Distance = 1) and Cooldown $\\le$ 0:
    * Deals damage based on mob type multiplied by `task.mob_damage_multiplier`.
    * **Sleep Critical:** Damage is multiplied by **2.5x** if the player is sleeping. Wakes the player.
    * Sets Cooldown to 5.
* **Despawn:** Despawns if the distance to the player exceeds `params.mob_despawn_distance` (unless fighting a Boss).

**2. Passive Mobs (Cows, Sheep)**
* **AI Logic:** Moves randomly (50% chance to stay still, 50% chance to move).
* **Despawn:** Same distance rules as Melee Mobs.

**3. Ranged Mobs (Archers)**
* **AI Logic (Kiting Behavior):**
    * **Distance $\\le$ 3:** Moves **AWAY** from player.
    * **Distance $\\ge$ 6:** Moves **TOWARD** player.
    * **Distance 4-5:** Random movement.
    * *Noise:* 15% chance to ignore logic and move randomly.
* **Attack:** Fires a projectile if:
    * Distance is **4 or 5**.
    * **OR** Distance is $\\le$ 3 AND the mob is cornered (cannot move away).
* **Cooldown:** Sets to 4 after shooting.



**4. Mob Projectiles**
* **Movement:** Travels in a straight line.
* **Collision:**
    * **Player:** Deals damage and destroys the projectile.
    * **Walls:** Destroys the projectile (Arrows travel over Water).
    * **Infrastructure:** Destroys **Furnaces** and **Crafting Tables** upon impact.

**5. Player Projectiles**
* **Damage Calculation:**
    * **Base:** Defined by projectile type (Arrow, Fireball, Iceball).
    * **Enchantment:** Bow enchantments add elemental damage to arrows.
    * **Attribute Scaling:**
        * **Arrows:** Multiplied by $1 + 0.2 \times (\text{Dexterity} - 1)$.
        * **Magic:** Multiplied by $1 + 0.5 \times (\text{Intelligence} - 1)$.
* **Collision:** checks for impact with mobs (via `attack_mob`) or walls along the trajectory.

The `spawn_mobs` function manages the stochastic generation of new entities on the current level. It evaluates spawning for Passive, Melee, and Ranged mobs sequentially.

**Global Spawn Coefficient:**
Spawn probabilities are scaled dynamically:
1.  **Uncleared Bonus:** If the level is not yet cleared (kills < required), spawn rates are **tripled (3x)**.
2.  **Boss Wave:** During specific "Boss Spawn Waves" in the Necromancer fight, rates are multiplied by **1000x** to guarantee spawns.

**General Spawn Constraints:**
* **Cap:** Spawning fails if the count of that mob type reaches its specific `max_[type]_mobs` limit.
* **Collision:** Mobs cannot spawn on top of existing mobs or solid blocks.
* **Despawn Radius:** All spawns must occur within `params.mob_despawn_distance` of the player.

**Spawn Logic by Type:**

| Feature | Passive Mobs | Melee Mobs (Hostile) | Ranged Mobs (Hostile) |
| :--- | :--- | :--- | :--- |
| **Spawn Chance** | Base Chance $\times$ `task.passive_spawn_multiplier` | (Base + **Night Bonus**) $\times$ `task.melee_spawn_multiplier` | Base Chance $\times$ `task.ranged_spawn_multiplier` |
| **Night Bonus** | None | Increases as `light_level` decreases (Square of darkness). | None |
| **Valid Terrain** | Grass, Path, Fire/Ice Grass | **Normal:** Any valid ground.<br>**Boss:** Graves only. | **Normal:** Any valid ground.<br>**Boss:** Graves only.<br>**Deep Thing:** Water only. |
| **Min Distance** | $> 3$ blocks from player | **Normal:** $> 9$ blocks.<br>**Boss:** $\\le 6$ blocks. | **Normal:** $> 9$ blocks.<br>**Boss:** $\\le 6$ blocks. |
| **Boss Logic** | **DISABLED** during boss fights. | Spawns heavily on Graves. | Spawns heavily on Graves. |



**Entity Initialization:**
* **Type:** Determined by `player_level` (or `boss_progress` during boss fights).
* **Health:** Base health from `MOB_TYPE_HEALTH_MAPPING` multiplied by `task.mob_health_multiplier`.
* **Position:** Randomly selected from valid tiles meeting all criteria.

The `update_plants` function manages the passive growth cycle of farming crops.

**Growth Logic:**
* **Aging:** Every timestep, the age of all active plants is incremented by 1.
* **Maturation:** When a plant's age reaches the threshold defined by `task.growing_plants_age`, it transitions to a mature state.

**State Update:**
* **Visual/Functional Change:** Upon maturation, the block at the plant's coordinates on Map Level 0 is updated from `PLANT` (unripe) to `RIPE_PLANT` (harvestable).

The `update_player_intrinsics` function manages the agent's biological metabolism, handling the decay and regeneration of stats via an **Accumulator/Threshold system**. Hidden counters increment every step; when they overflow a threshold, the actual inventory/stat updates.

**Global Modifiers & Task Parameters:**
* **Dexterity:** Reduces the base rate of Hunger, Thirst, and Fatigue accumulation (`intrinsic_decay_coeff`).
* **Task Multipliers:**
    * `task.needs_depletion_multiplier`: Scales the speed at which Hunger, Thirst, and Fatigue accumulate.
    * `task.health_recover_multiplier`: Scales how fast Health restores when needs are met.
    * `task.health_loss_multiplier`: Scales how fast Health drains when starving/dehydrated.
    * `task.mana_recover_multiplier`: Scales the specific Mana regeneration bonus granted by **Intelligence**.

**States:**
* **Sleeping (`Action.SLEEP`):** Starts if `Energy < Max`. Wakes when `Energy == Max`. Doubles health recovery speed and halves hunger/thirst accumulation.
* **Resting (`Action.REST`):** Starts if `Health < Max`. Wakes if `Health == Max` OR `Food/Drink` run out.

**Survival Mechanics:**

| Stat | Accumulation Logic | Threshold | Result |
| :--- | :--- | :--- | :--- |
| **Hunger** | (Base Rate based on Dex) $\times$ `task.needs_depletion_multiplier` | **> 25** | **-1 Food** |
| **Thirst** | (Base Rate based on Dex) $\times$ `task.needs_depletion_multiplier` | **> 20** | **-1 Drink** |
| **Fatigue** | (Base Rate based on Dex) $\times$ `task.needs_depletion_multiplier` | **> 30**<br>**< -10** | **-1 Energy**<br>**+1 Energy** |

**Regeneration Logic:**

| Stat | Conditions | Calculation | Threshold | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Health** | **Requirements:** Food, Drink, & Energy > 0.<br>**If Met:** | Rate = (2.0 if Sleeping, else 1.0) $\times$ `task.health_recover_multiplier` | **> 25** | **+1 Health** |
| | **If Failed (Starving):** | Rate = (-0.5 if Sleeping, else -1.0) $\times$ `task.health_loss_multiplier` | **< -15** | **-1 Health** |
| **Mana** | Always active. | Rate = (Base + Sleep Bonus) $\times$ (1 + 0.25 $\times$ (Int - 1) $\times$ `task.mana_recover_multiplier`) | **> 30** | **+1 Mana** |
""".strip()
