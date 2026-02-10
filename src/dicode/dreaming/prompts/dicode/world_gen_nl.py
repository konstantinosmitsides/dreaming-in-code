context = """
# Craftax Environment: Initial State Description

## 1. World Structure & Geography
The world consists of 9 distinct levels vertically stacked (z-axis). The player navigates between levels using ladders (Ladder Down / Ladder Up).

### Level Ordering (Index 0 to 8):
0.  **Overworld (SmoothGen):**
    * Terrain: Grass (Default), Water (Sea), Sand (Coast), Stone (Mountains).
    * Features: Trees, Lava lakes.
    * Ores: Coal (3%), Iron (2%), Diamond (0.1%).
    * Lighting: Fully lit (1.0).
1.  **Dungeon:**
    * Structure: Procedurally generated rooms connected by corridors.
    * Features: Chests, Fountains, Torches in corners.
    * Special: Basic pathing and walls.
2.  **Gnomish Mines (SmoothGen):**
    * Terrain: Path/Stone mix.
    * Features: Stalagmites (Trees), Lava lakes.
    * Ores: Coal, Iron, Diamond, Sapphire, Ruby.
    * Lighting: Pitch black (0.0).
3.  **Sewers (Dungeon):**
    * Structure: Rooms and corridors.
    * Features: Water channels, Ice Enchantment Tables.
4.  **Vaults (Dungeon):**
    * Structure: Rooms and corridors.
    * Features: Fire Enchantment Tables.
5.  **Troll Mines (SmoothGen):**
    * Terrain: Similar to Gnomish Mines but different mob density/generation.
    * Ores: Higher probabilities for Iron and Gems.
    * Lighting: Pitch black (0.0).
6.  **Fire Level (SmoothGen):**
    * Terrain: Fire Grass, Lava Oceans.
    * Features: Fire Trees.
    * Ores: Coal, Ruby.
    * Lighting: Fully lit (1.0).
7.  **Ice Level (SmoothGen):**
    * Terrain: Ice Grass, Water Oceans.
    * Features: Ice Shrubs.
    * Ores: Diamond, Sapphire.
    * Lighting: Pitch black (0.0).
8.  **Boss Level (SmoothGen):**
    * Terrain: Surrounded by Walls.
    * Features: Graves, Necromancer Spawn.
    * Ores: Mossy Walls, Grave Variants.
    * Lighting: Pitch black (0.0).

## 2. Terrain Generation Logic
* **SmoothGen Levels (0, 2, 5, 6, 7, 8):** Generated using fractal noise (Perlin-like). Terrain height determines Water -> Sand -> Grass -> Mountain -> Inner Cave. Ores are distributed stochastically within specific blocks (usually Stone).
* **Dungeon Levels (1, 3, 4):** Generated using a room-placement algorithm with collision detection. Rooms are connected via orthogonal paths. Special blocks (Chests, Fountains) are placed randomly within rooms. Walls adjacently touching paths become "Mossy Walls" with low probability.

## 3. Player Initial State
* **Position:** Center of the Overworld (Level 0).
* **Attributes:**
    * Strength: 1
    * Dexterity: 1
    * Intelligence: 1
* **Status Bars (Max 9):**
    * Health: 9.0 / 9.0
    * Food (Hunger): 9 / 9
    * Drink (Thirst): 9 / 9
    * Energy (Fatigue): 9 / 9
    * Mana: 9 / 9
* **Inventory:** Completely empty (all counts set to 0).
    * *Note:* If `god_mode` is True, inventory starts full (99 of all resources, high-tier tools).
* **Equipment:**
    * Sword/Bow Enchantment: Level 0.
    * Armor Enchantment: [0, 0, 0, 0].
* **Active Effects:**
    * Is Sleeping: False
    * Is Resting: False
    * Learned Spells: [False, False]

## 4. World Dynamics & Mobs
* **Mobs:** Arrays are initialized for Melee, Ranged, and Passive mobs, but start masked (inactive) until spawned by game logic.
* **Projectiles:** Arrays initialized for Player and Mob projectiles (inactive).
* **Plants:** Growing plants array initialized to zero.
* **Potions:** The color-to-effect mapping for potions is randomized (permuted) at the start of every episode (6 types).
* **Ladders:**
    * Ladders are procedurally placed.
    * The Ladder Down on Level 0 starts OPEN (Logic: `monsters_killed[0]` is initialized to 10 to bypass the kill requirement for the first ladder).
* **Achievements:** All set to False.
* **Boss:** Progress set to 0.

## 5. Global Constants
* **Map Size:** Defined by `static_params`.
* **Chunk Size:** 16 (for dungeon generation).
* **Light Calculation:** Recalculated based on torches, lava proximity, and level default light settings.
""".strip()
