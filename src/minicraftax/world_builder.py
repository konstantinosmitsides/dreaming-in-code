import jax
import jax.numpy as jnp
import jax.scipy as jsp


# Import from the full craftax game
from craftax.craftax.constants import (
	MOB_TYPE_HEALTH_MAPPING,
	TORCH_LIGHT_MAP,
	Achievement,
	Action,
	BlockType,
	ItemType,
	MobType,
)
from craftax.craftax.craftax_state import EnvParams, Mobs, StaticEnvParams
from craftax.craftax.game_logic import calculate_light_level, get_distance_map
from craftax.craftax.util.noise import generate_fractal_noise_2d
from craftax.craftax.world_gen.world_gen import get_new_empty_inventory, get_new_full_inventory
from craftax.craftax.world_gen.world_gen_configs import (
	ALL_DUNGEON_CONFIGS,
	ALL_SMOOTHGEN_CONFIGS,
	DungeonConfig,
	SmoothGenConfig,
)

from minicraftax.craftax_state import EnvState


def _generate_base_smoothworld_level(
	rng: jax.Array,
	static_params: StaticEnvParams,
	player_position: jax.Array,
	config: SmoothGenConfig,
	params: EnvParams,
) -> jnp.ndarray:
	"""Generates a single base terrain level using fractal noise and distance maps.

	Excludes ore resources which are added in a separate pass.

	Args:
		rng: JAX random number generator key.
		static_params: Static environment parameters.
		player_position: The player's current position to determine spawn safety.
		config: Configuration for smooth generation (biomes, thresholds).
		params: Dynamic environment parameters.

	Returns:
		tuple: (new_map, item_map, light_map, ladder_down, ladder_up)
	"""
	# 1. Calculate player proximity maps
	player_proximity_map = get_distance_map(player_position, static_params.map_size).astype(
		jnp.float32
	)
	player_proximity_map_water = jnp.clip(
		player_proximity_map / config.player_proximity_map_water_strength,
		0.0,
		config.player_proximity_map_water_max,
	)
	player_proximity_map_mountain = jnp.clip(
		player_proximity_map / config.player_proximity_map_mountain_strength,
		0.0,
		config.player_proximity_map_mountain_max,
	)

	# 2. Define noise resolutions
	larger_res = (static_params.map_size[0] // 4, static_params.map_size[1] // 4)
	small_res = (static_params.map_size[0] // 16, static_params.map_size[1] // 16)
	x_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 2)

	# 3. Generate base noise maps
	rng, water_rng, mountain_rng, path_rng, tree_rng = jax.random.split(rng, 5)
	water = (
		generate_fractal_noise_2d(
			water_rng,
			static_params.map_size,
			small_res,
			octaves=1,
			override_angles=params.fractal_noise_angles[0],
		)
		+ player_proximity_map_water
		- 1.0
	)
	mountain = (
		generate_fractal_noise_2d(
			mountain_rng,
			static_params.map_size,
			small_res,
			octaves=1,
			override_angles=params.fractal_noise_angles[1],
		)
		+ 0.05
		+ player_proximity_map_mountain
		- 1.0
	)
	path_x = generate_fractal_noise_2d(
		path_rng,
		static_params.map_size,
		x_res,
		octaves=1,
		override_angles=params.fractal_noise_angles[2],
	)
	tree_noise = generate_fractal_noise_2d(
		tree_rng,
		static_params.map_size,
		larger_res,
		octaves=1,
		override_angles=params.fractal_noise_angles[3],
	)

	# 4. Progressively build the map from noise
	new_map = jnp.full(static_params.map_size, config.default_block)
	new_map = jnp.where(water > config.water_threshold, config.sea_block, new_map)
	sand_map = jnp.logical_and(water > config.sand_threshold, new_map != config.sea_block)
	new_map = jnp.where(sand_map, config.coast_block, new_map)
	new_map = jnp.where(mountain > 0.7, config.mountain_block, new_map)
	path = jnp.logical_and(mountain > 0.7, path_x > 0.8)
	new_map = jnp.where(path > 0.5, config.path_block, new_map)
	path_y = path_x.T
	path = jnp.logical_and(mountain > 0.7, path_y > 0.8)
	new_map = jnp.where(path > 0.5, config.path_block, new_map)
	caves = jnp.logical_and(mountain > 0.85, water > 0.4)
	new_map = jnp.where(caves > 0.5, config.inner_mountain_block, new_map)

	# 5. Add Trees
	rng, tree_placement_rng = jax.random.split(rng)
	tree = (tree_noise > config.tree_threshold_perlin) * jax.random.uniform(
		tree_placement_rng, shape=static_params.map_size
	) > config.tree_threshold_uniform
	tree = jnp.logical_and(tree, new_map == config.tree_requirement_block)
	new_map = jnp.where(tree, config.tree, new_map)

	# Add Ores
	def _add_ore(carry, index):
		rng, map = carry
		rng, _rng = jax.random.split(rng)
		ore_map = jnp.logical_and(
			map == config.ore_requirement_blocks[index],
			jax.random.uniform(_rng, static_params.map_size) < config.ore_chances[index],
		)
		map = jnp.where(ore_map, config.ores[index], map)

		return (rng, map), None

	rng, _rng = jax.random.split(rng)
	(_, new_map), _ = jax.lax.scan(_add_ore, (_rng, new_map), jnp.arange(5))

	# 6. Add Lava
	lava_map = jnp.logical_and(mountain > 0.85, tree_noise > 0.7)
	new_map = jnp.where(lava_map, config.lava, new_map)

	# Add diamond if always_diamond flag is set
	adding_diamond = jnp.logical_and(
		config.default_block == BlockType.GRASS.value,  # Hacky check for overworld
		params.always_diamond,
	)
	valid_diamond = (new_map.flatten() == BlockType.STONE.value).astype(jnp.float32)
	rng, _rng = jax.random.split(rng)
	diamond_index = jax.random.choice(
		_rng,
		jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
		p=valid_diamond / valid_diamond.sum(),
	)
	diamond_position = jnp.array(
		[
			diamond_index // static_params.map_size[0],
			diamond_index % static_params.map_size[0],
		]
	)
	diamond_replace_block = jax.lax.select(
		adding_diamond, BlockType.DIAMOND.value, BlockType.STONE.value
	)
	new_map = new_map.at[diamond_position[0], diamond_position[1]].set(diamond_replace_block)

	# Light map
	light_map = jnp.ones(static_params.map_size, dtype=jnp.float32) * config.default_light

	# Make sure player spawns on grass
	new_map = new_map.at[player_position[0], player_position[1]].set(config.player_spawn)

	item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)

	valid_ladder_down = new_map.flatten() == config.valid_ladder
	rng, _rng = jax.random.split(rng)
	ladder_index = jax.random.choice(
		_rng,
		jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
		p=valid_ladder_down,
	)
	ladder_down = jnp.array(
		[
			ladder_index // static_params.map_size[0],
			ladder_index % static_params.map_size[0],
		]
	)

	item_map = item_map.at[ladder_down[0], ladder_down[1]].set(
		ItemType.LADDER_DOWN.value * config.ladder_down
		+ item_map[ladder_down[0], ladder_down[1]] * (1 - config.ladder_down)
	)

	valid_ladder_up = new_map.flatten() == config.valid_ladder
	rng, _rng = jax.random.split(rng)
	ladder_index = jax.random.choice(
		_rng,
		jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
		p=valid_ladder_up,
	)
	ladder_up = jnp.array(
		[
			ladder_index // static_params.map_size[0],
			ladder_index % static_params.map_size[0],
		]
	)

	LIGHT_MAP_AROUND_LADDER = TORCH_LIGHT_MAP * (
		1 - config.default_light
	) + config.default_light * jnp.ones((9, 9))

	light_map = jax.lax.dynamic_update_slice(
		light_map, LIGHT_MAP_AROUND_LADDER, ladder_up - jnp.array([4, 4])
	)

	z = jnp.array([[0.2, 0.7, 0.2], [0.7, 1, 0.7], [0.2, 0.7, 0.2]]) * (
		config.lava == BlockType.LAVA.value
	)
	light_map += jsp.signal.convolve(lava_map, z, mode="same")
	light_map = jnp.clip(light_map, 0.0, 1.0)

	item_map = item_map.at[ladder_up[0], ladder_up[1]].set(
		ItemType.LADDER_UP.value * config.ladder_up
		+ item_map[ladder_up[0], ladder_up[1]] * (1 - config.ladder_up)
	)

	return new_map, item_map, light_map, ladder_down, ladder_up


def _generate_base_dungeon_level(
	rng: jax.Array, static_params: StaticEnvParams, config: DungeonConfig
) -> jnp.ndarray:
	"""Generates a procedural dungeon level with rooms and corridors.

	Uses a grid-based approach to place rooms and connects them with paths.
	Also places special items like chests, fountains, and torches.

	Args:
		rng: JAX random number generator key.
		static_params: Static environment parameters.
		config: Configuration for dungeon generation.

	Returns:
		tuple: (map, item_map, light_map, ladder_down, ladder_up)
	"""
	# 1. Setup
	chunk_size = 16
	world_chunk_width = static_params.map_size[0] // chunk_size
	world_chunk_height = static_params.map_size[1] // chunk_size
	room_occupancy_chunks = jnp.ones(world_chunk_width * world_chunk_height)
	num_rooms = 8
	min_room_size = 5
	max_room_size = 10

	rng, _rng = jax.random.split(rng)
	room_sizes = jax.random.randint(
		_rng, shape=(num_rooms, 2), minval=min_room_size, maxval=max_room_size
	)
	map = jnp.ones(static_params.map_size, dtype=jnp.int32) * BlockType.WALL.value
	padded_map = jnp.pad(map, max_room_size, constant_values=0)

	item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)
	padded_item_map = jnp.pad(item_map, max_room_size, constant_values=0)

	# 2. Add Rooms
	def _add_room(carry, room_index):
		block_map, item_map, room_occupancy_chunks, rng = carry
		rng, _rng = jax.random.split(rng)
		room_chunk = jax.random.choice(
			_rng, jnp.arange(world_chunk_width * world_chunk_height), p=room_occupancy_chunks
		)
		room_occupancy_chunks = room_occupancy_chunks.at[room_chunk].set(0)
		room_position = jnp.array(
			[
				(room_chunk % world_chunk_height) * chunk_size,
				(room_chunk // world_chunk_height) * chunk_size,
			]
		) + jnp.array([max_room_size, max_room_size])
		rng, _rng = jax.random.split(rng)
		room_position += jax.random.randint(_rng, (2,), minval=0, maxval=chunk_size - min_room_size)
		slice = jax.lax.dynamic_slice(block_map, room_position, (max_room_size, max_room_size))
		xs = jnp.expand_dims(jnp.arange(max_room_size), axis=-1).repeat(max_room_size, axis=-1)
		ys = jnp.expand_dims(jnp.arange(max_room_size), axis=0).repeat(max_room_size, axis=0)
		room_mask = jnp.logical_and(xs < room_sizes[room_index, 0], ys < room_sizes[room_index, 1])
		slice = room_mask * BlockType.PATH.value + (1 - room_mask) * slice
		block_map = jax.lax.dynamic_update_slice(block_map, slice, room_position)

		# Torches in corner
		item_map = item_map.at[room_position[0], room_position[1]].set(ItemType.TORCH.value)
		item_map = item_map.at[
			room_position[0] + room_sizes[room_index, 0] - 1, room_position[1]
		].set(ItemType.TORCH.value)
		item_map = item_map.at[
			room_position[0], room_position[1] + room_sizes[room_index, 1] - 1
		].set(ItemType.TORCH.value)
		item_map = item_map.at[
			room_position[0] + room_sizes[room_index, 0] - 1,
			room_position[1] + room_sizes[room_index, 1] - 1,
		].set(ItemType.TORCH.value)

		# Chest
		rng, _rng = jax.random.split(rng)
		chest_position = jax.random.randint(
			_rng,
			shape=(2,),
			minval=jnp.ones(2),
			maxval=room_sizes[room_index] - jnp.ones(2),
		)
		block_map = block_map.at[
			room_position[0] + chest_position[0], room_position[1] + chest_position[1]
		].set(BlockType.CHEST.value)

		# Fountain
		rng, _rng, __rng = jax.random.split(rng, 3)
		fountain_position = jax.random.randint(
			_rng,
			shape=(2,),
			minval=jnp.ones(2),
			maxval=room_sizes[room_index] - jnp.ones(2),
		)
		room_has_fountain = jax.random.uniform(__rng) > 0.5
		fountain_block = (
			room_has_fountain * config.fountain_block
			+ (1 - room_has_fountain)
			* block_map[
				room_position[0] + fountain_position[0],
				room_position[1] + fountain_position[1],
			]
		)
		block_map = block_map.at[
			room_position[0] + fountain_position[0],
			room_position[1] + fountain_position[1],
		].set(fountain_block)

		return (block_map, item_map, room_occupancy_chunks, rng), room_position

	rng, _rng = jax.random.split(rng)
	(padded_map, padded_item_map, _, _), room_positions = jax.lax.scan(
		_add_room,
		(padded_map, padded_item_map, room_occupancy_chunks, _rng),
		jnp.arange(num_rooms),
	)

	def _add_path(carry, path_index):
		cmap, included_rooms_mask, rng = carry

		path_source = room_positions[path_index]

		rng, _rng = jax.random.split(rng)
		sink_index = jax.random.choice(_rng, jnp.arange(num_rooms), p=included_rooms_mask)
		path_sink = room_positions[sink_index]

		# Horizontal component
		entire_row = cmap[path_source[0]]
		path_indexes = jnp.arange(static_params.map_size[0] + 2 * max_room_size)
		path_indexes = path_indexes - path_source[1]
		horizontal_distance = path_sink[1] - path_source[1]
		path_indexes = path_indexes * jnp.sign(horizontal_distance)

		horizontal_mask = jnp.logical_and(
			path_indexes >= 0, path_indexes <= jnp.abs(horizontal_distance)
		)
		horizontal_mask = jnp.logical_and(horizontal_mask, jnp.sign(horizontal_distance))
		horizontal_mask = jnp.logical_and(horizontal_mask, entire_row == BlockType.WALL.value)

		new_row = horizontal_mask * BlockType.PATH.value + (1 - horizontal_mask) * entire_row

		cmap = jax.lax.dynamic_update_slice(
			cmap,
			jnp.expand_dims(new_row, axis=0),
			path_source,
		)

		# Vertical component
		entire_col = cmap[:, path_sink[1]]
		path_indexes = jnp.arange(static_params.map_size[1] + 2 * max_room_size)
		path_indexes = path_indexes - path_source[0]
		vertical_distance = path_sink[0] - path_source[0]
		path_indexes = path_indexes * jnp.sign(vertical_distance)

		vertical_mask = jnp.logical_and(
			path_indexes >= 0, path_indexes <= jnp.abs(vertical_distance)
		)
		vertical_mask = jnp.logical_and(vertical_mask, jnp.sign(vertical_distance))

		vertical_mask = jnp.logical_and(vertical_mask, entire_col == BlockType.WALL.value)

		new_col = vertical_mask * BlockType.PATH.value + (1 - vertical_mask) * entire_col

		cmap = jax.lax.dynamic_update_slice(
			cmap,
			jnp.expand_dims(new_col, axis=-1),
			path_sink,
		)

		rng, _rng = jax.random.split(rng)
		included_rooms_mask = included_rooms_mask.at[path_index].set(True)
		return (cmap, included_rooms_mask, _rng), None

	rng, _rng = jax.random.split(rng)
	included_rooms_mask = jnp.zeros(num_rooms, dtype=bool).at[-1].set(True)
	(
		(padded_map, _, _),
		_,
	) = jax.lax.scan(_add_path, (padded_map, included_rooms_mask, _rng), jnp.arange(0, num_rooms))

	# Place special block in a random room
	special_block_position = room_positions[0] + jnp.array([2, 2])
	padded_map = padded_map.at[special_block_position[0], special_block_position[1]].set(
		config.special_block
	)

	map = padded_map[max_room_size:-max_room_size, max_room_size:-max_room_size]
	item_map = padded_item_map[max_room_size:-max_room_size, max_room_size:-max_room_size]

	# Visual stuff
	c_path_map = map != BlockType.WALL.value
	z = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
	adj_path_map = jsp.signal.convolve(c_path_map, z, mode="same")
	adj_path_map = adj_path_map > 0.5

	rng, _rng = jax.random.split(rng)
	rare_map = jax.random.choice(
		_rng,
		jnp.array([False, True]),
		static_params.map_size,
		p=jnp.array([0.9, 0.1]),
	)

	wall_map = rare_map * BlockType.WALL_MOSS.value + (1 - rare_map) * BlockType.WALL.value

	rare_map = jnp.logical_and(rare_map, map == BlockType.PATH.value)
	rare_map = jnp.logical_and(rare_map, item_map == ItemType.NONE.value)
	path_map = rare_map * config.rare_path_replacement_block + (1 - rare_map) * map

	is_wall_map = jnp.logical_and(map == BlockType.WALL.value, adj_path_map)
	is_darkness_map = jnp.logical_not(adj_path_map)
	is_path_map = jnp.logical_not(jnp.logical_or(is_wall_map, is_darkness_map))

	map = (
		is_path_map * path_map + is_wall_map * wall_map + is_darkness_map * BlockType.DARKNESS.value
	)

	light_map = jnp.ones(static_params.map_size, dtype=jnp.float32)

	# Ladders
	valid_ladder_down = (map.flatten() == BlockType.PATH.value).astype(jnp.float32)
	rng, _rng = jax.random.split(rng)
	ladder_index = jax.random.choice(
		_rng,
		jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
		p=valid_ladder_down / valid_ladder_down.sum(),
	)
	ladder_down_position = jnp.array(
		[
			ladder_index // static_params.map_size[0],
			ladder_index % static_params.map_size[0],
		]
	)

	item_map = item_map.at[ladder_down_position[0], ladder_down_position[1]].set(
		ItemType.LADDER_DOWN.value
	)

	valid_ladder_up = map.flatten() == BlockType.PATH.value
	rng, _rng = jax.random.split(rng)
	ladder_index = jax.random.choice(
		_rng,
		jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
		p=valid_ladder_up,
	)
	ladder_up_position = jnp.array(
		[
			ladder_index // static_params.map_size[0],
			ladder_index % static_params.map_size[0],
		]
	)
	item_map = item_map.at[ladder_up_position[0], ladder_up_position[1]].set(
		ItemType.LADDER_UP.value
	)

	return map, item_map, light_map, ladder_down_position, ladder_up_position


def _place_item_randomly_pure_jit_safe(
	rng: jax.Array,
	current_item_map_slice: jax.Array,  # The item_map level to modify
	underlying_map_slice: jax.Array,  # The corresponding map level to check conditions
	item_type_value: int,  # The item value to place (e.g., ItemType.TORCH.value)
	n: int,  # Number of items to place
	on_block_values: jax.Array,  # Valid block types on the underlying map
) -> jax.Array:
	"""Places `n` items at random valid locations on an item map.

	This function is pure and JIT-safe. It checks validity against the underlying block map
	and ensures items are only placed in empty slots.

	Args:
		rng: JAX random key.
		current_item_map_slice: 2D array of current items.
		underlying_map_slice: 2D array of underlying blocks.
		item_type_value: Integer value of the item to place.
		n: Number of items to place.
		on_block_values: Array of valid block types to place items on.

	Returns:
		Updated item map slice.
	"""
	# 1. Create a boolean mask of all valid placement locations.
	# Condition 1: Underlying block must be valid.
	valid_underlying_block_mask = jnp.isin(underlying_map_slice, on_block_values)
	# Condition 2: The spot on the item map must be empty (assuming ItemType.NONE.value is 0).
	item_slot_is_empty_mask = current_item_map_slice == ItemType.NONE.value
	# Combine conditions.
	valid_mask = valid_underlying_block_mask & item_slot_is_empty_mask

	# 2. Assign a random score (logit) to every cell.
	random_logits = jax.random.uniform(rng, shape=current_item_map_slice.shape)

	# 3. Mask out invalid locations by giving them a terrible score (-infinity).
	masked_logits = jnp.where(valid_mask, random_logits, -jnp.inf)

	# 4. Find the flat indices of the `n` locations with the HIGHEST random scores.
	flat_indices = jnp.argsort(masked_logits.flatten())
	# Ensure n does not exceed the total number of cells, prevents out-of-bounds in argsort slicing
	safe_n = jnp.minimum(n, current_item_map_slice.size)
	selected_flat_indices = flat_indices[-safe_n:]  # Use safe_n

	# 5. Convert the selected 1D indices back to 2D coordinates.
	map_width = current_item_map_slice.shape[1]
	rows = selected_flat_indices // map_width
	cols = selected_flat_indices % map_width

	# 6. Final check: Ensure selected spots were actually valid (handles n > num_valid_locations).
	is_selection_valid = valid_mask[rows, cols]

	# 7. Determine the value to set: the new item or the original item value (NONE) if invalid.
	value_to_set = jnp.where(
		is_selection_valid,
		item_type_value,
		current_item_map_slice[rows, cols],  # Should be ItemType.NONE.value
	)

	# 8. Perform the update on the item map slice.
	new_item_map_slice = current_item_map_slice.at[rows, cols].set(value_to_set)

	return new_item_map_slice


def _place_item_randomly_near_pure(
	rng: jax.Array,
	current_item_map_slice: jax.Array,  # The item_map level to modify
	underlying_map_slice: jax.Array,  # The corresponding map level to check conditions
	map_size: tuple[int, int],  # Shape of the map slice (height, width)
	item_type_value: int,  # The item value to place
	n: int,  # Number of items to place
	target_pos: tuple[int, int],  # The (row, col) to place near
	min_dist: int,  # Minimum Manhattan distance from target_pos
	max_dist: int,  # Maximum Manhattan distance from target_pos
	on_block_values: jax.Array,  # Valid block types on the underlying map
) -> jax.Array:
	"""A pure, JIT-safe function to place n items near a target on an item map,
	checking conditions against the underlying block map.
	"""
	# 1. Create masks.
	# Distance Mask:
	rows_grid, cols_grid = jnp.indices(map_size)
	dist_map = jnp.abs(rows_grid - target_pos[0]) + jnp.abs(cols_grid - target_pos[1])
	distance_mask = (dist_map >= min_dist) & (dist_map <= max_dist)
	# Underlying Block Mask:
	valid_underlying_block_mask = jnp.isin(underlying_map_slice, on_block_values)
	# Empty Item Slot Mask:
	item_slot_is_empty_mask = current_item_map_slice == ItemType.NONE.value
	# Combine all conditions.
	final_mask = distance_mask & valid_underlying_block_mask & item_slot_is_empty_mask

	# 2. Assign a random score to every cell.
	random_logits = jax.random.uniform(rng, shape=map_size)

	# 3. Mask out invalid locations by giving them a terrible score (-infinity).
	masked_logits = jnp.where(final_mask, random_logits, -jnp.inf)

	# 4. Find the flat indices of the `n` locations with the HIGHEST scores.
	flat_indices = jnp.argsort(masked_logits.flatten())
	# Ensure n does not exceed the total number of cells
	safe_n = jnp.minimum(n, current_item_map_slice.size)
	selected_flat_indices = flat_indices[-safe_n:]  # Use safe_n

	# 5. Convert the selected 1D indices back to 2D coordinates.
	map_width = map_size[1]
	rows = selected_flat_indices // map_width
	cols = selected_flat_indices % map_width

	# 6. Final check: Ensure selected spots were actually valid.
	is_selection_valid = final_mask[rows, cols]

	# 7. Determine the value to set: the new item or the original item value (NONE).
	value_to_set = jnp.where(
		is_selection_valid,
		item_type_value,
		current_item_map_slice[rows, cols],  # Should be ItemType.NONE.value
	)

	# 8. Perform the update on the item map slice.
	new_item_map_slice = current_item_map_slice.at[rows, cols].set(value_to_set)

	return new_item_map_slice


def _place_randomly_pure_jit_safe(
	rng: jax.Array,
	current_map: jax.Array,
	block_type_value: int,
	n: int,
	on_block_values: jax.Array,
) -> jax.Array:
	"""A pure, JIT-safe function to place n blocks at random valid locations."""
	# 1. Create a boolean mask of all valid placement locations.
	valid_mask = jnp.isin(current_map, on_block_values)

	# 2. Assign a random score (logit) to every cell in the entire map.
	random_logits = jax.random.uniform(rng, shape=current_map.shape)

	# 3. Mask out invalid locations by giving them a terrible score (-infinity).
	# This ensures they will never be chosen as one of the "best" random spots.
	masked_logits = jnp.where(valid_mask, random_logits, -jnp.inf)

	# 4. Find the flat indices of the `n` locations with the HIGHEST random scores.
	# We use argsort and take the last `n` elements, as argsort sorts ascending.
	# The shape of the output is always (n,), which is static and JIT-friendly.
	flat_indices = jnp.argsort(masked_logits.flatten())
	selected_flat_indices = flat_indices[-n:]

	# 5. Convert the selected 1D indices back to 2D coordinates.
	map_width = current_map.shape[1]
	rows = selected_flat_indices // map_width
	cols = selected_flat_indices % map_width

	# 6. Edge Case: What if n > num_valid_locations?
	# Some selected indices will point to locations with a score of -inf.
	# We must ensure we only place blocks on locations that were actually valid.
	# We create a final check to see if the chosen spots are in the original valid_mask.
	is_selection_valid = valid_mask[rows, cols]

	# The value to write: either the new block or the original block if the spot was invalid.
	value_to_set = jnp.where(is_selection_valid, block_type_value, current_map[rows, cols])

	# 7. Perform the update. This now correctly handles all edge cases.
	new_map = current_map.at[rows, cols].set(value_to_set)

	return new_map


def _place_randomly_near_pure(
	rng: jax.Array,
	current_map: jax.Array,
	map_size: tuple[int, int],
	block_type_value: int,
	n: int,  # The number of blocks to place
	target_pos: tuple[int, int],
	min_dist: int,
	max_dist: int,
	on_block_values: jax.Array,
) -> jax.Array:
	"""A pure, JIT-safe function to place n blocks near a target."""
	# 1. Create distance and surface masks.
	rows_grid, cols_grid = jnp.indices(map_size)
	dist_map = jnp.abs(rows_grid - target_pos[0]) + jnp.abs(cols_grid - target_pos[1])
	distance_mask = (dist_map >= min_dist) & (dist_map <= max_dist)
	surface_mask = jnp.isin(current_map, on_block_values)
	final_mask = distance_mask & surface_mask

	# 2. Assign a random score to every cell.
	random_logits = jax.random.uniform(rng, shape=map_size)

	# 3. Invalidate cells outside the target area by giving them a losing score.
	masked_logits = jnp.where(final_mask, random_logits, -jnp.inf)

	# 4. Find the flat indices of the n locations with the HIGHEST scores.
	# This is the key change from argmax to argsort.
	flat_indices = jnp.argsort(masked_logits.flatten())
	selected_flat_indices = flat_indices[-n:]

	# 5. Convert the selected 1D indices back to 2D coordinates (now plural).
	rows = selected_flat_indices // map_size[1]
	cols = selected_flat_indices % map_size[1]

	# 6. Check which of the selected spots were actually valid.
	is_selection_valid = final_mask[rows, cols]

	# 7. Set the new value only where the selection was valid.
	value_to_set = jnp.where(is_selection_valid, block_type_value, current_map[rows, cols])

	# 8. Perform the batch update.
	new_map = current_map.at[rows, cols].set(value_to_set)

	return new_map


class WorldBuilder:
	"""A helper class to programmatically construct MiniCraftax worlds.

	Handles the generation of multi-level worlds, including terrain (smoothgen)
	and dungeons, as well as placing the player, mobs, and items.

	Attributes:
		static_params (StaticEnvParams): Static environment configuration.
		params (EnvParams): Dynamic environment configuration.
		map (jnp.ndarray): 3D array representing the world blocks (levels, H, W).
		item_map (jnp.ndarray): 3D array representing items.
		mob_map (jnp.ndarray): 3D array representing mob presence.
	"""

	def __init__(self, rng: jax.Array, static_params: StaticEnvParams, params: EnvParams):
		"""Initializes a blank multi-level world canvas."""
		self.static_params = static_params
		self.params = params

		# Initialize multi-level maps
		self.map = jnp.zeros((static_params.num_levels, *static_params.map_size), dtype=jnp.int32)
		self.item_map = jnp.zeros(
			(static_params.num_levels, *static_params.map_size), dtype=jnp.int32
		)
		self.mob_map = jnp.zeros((static_params.num_levels, *static_params.map_size), dtype=bool)

		# Player state
		self.player_position = jnp.array(
			[static_params.map_size[0] // 2, static_params.map_size[1] // 2], dtype=jnp.int32
		)
		self.player_level = 0
		self.player_direction = Action.UP.value

		self.inventory = jax.tree_util.tree_map(
			lambda x, y: jax.lax.select(params.god_mode, x, y),
			get_new_full_inventory(),
			get_new_empty_inventory(),
		)

		self.player_dexterity = 1
		self.player_strength = 1
		self.player_intelligence = 1

		self.sword_enchantment = 0
		self.bow_enchantment = 0
		self.armour_enchantments = jnp.zeros(4, dtype=jnp.int32)
		self.learned_spells = jnp.array([False, False], dtype=bool)

		self.monsters_killed = jnp.zeros(static_params.num_levels, dtype=jnp.int32).at[0].set(10)

		# Initialize multi-level mob structures
		def _generate_empty_mobs(max_mobs):
			return Mobs(
				position=jnp.zeros((static_params.num_levels, max_mobs, 2), dtype=jnp.int32),
				health=jnp.ones((static_params.num_levels, max_mobs), dtype=jnp.float32),
				mask=jnp.zeros((static_params.num_levels, max_mobs), dtype=bool),
				attack_cooldown=jnp.zeros((static_params.num_levels, max_mobs), dtype=jnp.int32),
				type_id=jnp.zeros((static_params.num_levels, max_mobs), dtype=jnp.int32),
			)

		self.melee_mobs = _generate_empty_mobs(static_params.max_melee_mobs)
		self.ranged_mobs = _generate_empty_mobs(static_params.max_ranged_mobs)
		self.passive_mobs = _generate_empty_mobs(static_params.max_passive_mobs)

		# Other state components
		self.growing_plants_positions = jnp.zeros(
			(static_params.max_growing_plants, 2), dtype=jnp.int32
		)
		self.growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
		self.growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

		self.generate_full_base_world(rng)

	def generate_full_base_world(self, rng: jax.Array):
		player_position = self.player_position

		# Generate smoothgens
		rngs = jax.random.split(rng, 7)
		rng, _rng = rngs[0], rngs[1:]

		smoothgens = jax.vmap(_generate_base_smoothworld_level, in_axes=(0, None, None, 0, None))(
			_rng, self.static_params, player_position, ALL_SMOOTHGEN_CONFIGS, self.params
		)

		# Generate dungeons
		rngs = jax.random.split(rng, 4)
		rng, _rng = rngs[0], rngs[1:]
		dungeons = jax.vmap(_generate_base_dungeon_level, in_axes=(0, None, 0))(
			_rng, self.static_params, ALL_DUNGEON_CONFIGS
		)

		# Splice smoothgens and dungeons in order of levels
		map, item_map, light_map, ladders_down, ladders_up = jax.tree_util.tree_map(
			lambda x, y: jnp.stack((x[0], y[0], x[1], y[1], y[2], x[2], x[3], x[4], x[5]), axis=0),
			smoothgens,
			dungeons,
		)

		# 3. Update the builder's state
		self.map, self.item_map, self.light_map, self.ladders_down, self.ladders_up = (
			map,
			item_map,
			light_map,
			ladders_down,
			ladders_up,
		)

		print("Generated full 9-level base world without ores.")
		return self

	def set_starting_floor(self, level: int):
		"""Sets the player's starting level in the multi-level world."""
		self.player_level = level

		# --- Calculate final player position ---
		# If starting level > 0, use the up_ladder position for that level.
		# Otherwise, use the currently set player_position (default or from set_player_start).
		self.player_position = jax.lax.cond(
			self.player_level > 0,
			lambda: self.ladders_up[
				self.player_level
			],  # Position on the 'up' ladder of the target level
			lambda: self.player_position,  # Default position (usually center of level 0)
		)
		return self

	def set_player_stats(
		self,
		dexterity: int = 1,
		strength: int = 1,
		intelligence: int = 1,
	):
		"""Sets the player's starting stats (dexterity, strength, intelligence).

		The values are clamped to a safe range [1, 5] to ensure validity.
		"""
		# Clamp values to the defined range [1, 5]
		self.player_dexterity = jnp.clip(dexterity, 1, 5).astype(jnp.int32)
		self.player_strength = jnp.clip(strength, 1, 5).astype(jnp.int32)
		self.player_intelligence = jnp.clip(intelligence, 1, 5).astype(jnp.int32)

		return self

	def set_player_inventory(self, inventory_dict: dict):
		"""Sets the player's starting inventory."""
		self.inventory = self.inventory.replace(**inventory_dict)
		return self

	def set_weapon_enchantments(self, sword: int = 0, bow: int = 0):
		"""Sets the player's starting weapon enchantments.
		- 0: No enchantment
		- 1: Fire enchantment
		- 2: Ice enchantment
		"""
		self.sword_enchantment = jnp.clip(sword, 0, 2).astype(jnp.int32)
		self.bow_enchantment = jnp.clip(bow, 0, 2).astype(jnp.int32)
		return self

	def set_armour_enchantments(
		self, helmet: int = 0, chestplate: int = 0, leggings: int = 0, boots: int = 0
	):
		"""Sets the player's starting armour enchantments.
		- 0: No enchantment
		- 1: Fire resistance
		- 2: Ice resistance
		"""
		enchantments = jnp.array(
			[
				jnp.clip(helmet, 0, 2),
				jnp.clip(chestplate, 0, 2),
				jnp.clip(leggings, 0, 2),
				jnp.clip(boots, 0, 2),
			],
			dtype=jnp.int32,
		)
		self.armour_enchantments = enchantments
		return self

	def set_learned_spells(self, fireball: bool = False, iceball: bool = False):
		"""Sets the spells the player has learned from the start."""
		self.learned_spells = jnp.array([fireball, iceball], dtype=bool)
		return self

	def set_monsters_killed(self, level: int, count: int):
		"""Sets the number of monsters considered killed on a specific level.

		This is a more granural mechanism for unlocking ladders to specific subsequent levels.
		The count is clamped to a non-negative value.
		"""
		# Ensure count is not negative and update the specific level
		safe_count = jnp.maximum(0, count).astype(jnp.int32)
		self.monsters_killed = self.monsters_killed.at[level].set(safe_count)
		return self

	def place_block(self, level: int, block_type: BlockType, position: tuple):
		"""Places a single block on a specific level of the map."""
		self.map = self.map.at[level, position[0], position[1]].set(block_type.value)
		return self

	def fill_area(self, level: int, block_type: BlockType, top_left: tuple, bottom_right: tuple):
		"""Fills a rectangular area on a specific level with a block type."""
		xx, yy = jnp.meshgrid(
			jnp.arange(self.static_params.map_size[0]),
			jnp.arange(self.static_params.map_size[1]),
			indexing="ij",
		)
		mask = (
			(xx >= top_left[0])
			& (xx <= bottom_right[0])
			& (yy >= top_left[1])
			& (yy <= bottom_right[1])
		)
		level_map = self.map[level]
		updated_level_map = jnp.where(mask, block_type.value, level_map)
		self.map = self.map.at[level].set(updated_level_map)
		return self

	def add_mob(
		self, level: int, mob_name: str, type_id: int, position: tuple, health: float = -1.0
	):
		"""UPDATED: Adds a single mob to a specific level with a specific type_id.

		Args:
		    level (int): The level to place the mob on.
		    mob_name (str): "melee", "ranged", or "passive".
		    type_id (int): The specific type from the mob enums (e.g., MeleeMobType.ZOMBIE.value).
		    position (tuple): The (row, col) coordinate.
		    health (float): Optional health, defaults to mob's max health.

		"""

		def _update_mob_array(mob_array, max_health):
			idx = jnp.argmin(mob_array.mask[level])
			final_health = jax.lax.select(health == -1.0, max_health, health)

			# Use .at[level, idx] to update the specific mob on the correct level
			new_pos = mob_array.position.at[level, idx].set(jnp.array(position, dtype=jnp.int32))
			new_health = mob_array.health.at[level, idx].set(final_health)
			new_mask = mob_array.mask.at[level, idx].set(True)
			new_type = mob_array.type_id.at[level, idx].set(type_id)

			return mob_array.replace(
				position=new_pos, health=new_health, mask=new_mask, type_id=new_type
			)

		# Select which mob array to update
		is_melee = mob_name == "melee"
		is_ranged = mob_name == "ranged"

		self.melee_mobs = jax.lax.cond(
			is_melee, lambda: _update_mob_array(self.melee_mobs, 5.0), lambda: self.melee_mobs
		)
		self.ranged_mobs = jax.lax.cond(
			is_ranged, lambda: _update_mob_array(self.ranged_mobs, 3.0), lambda: self.ranged_mobs
		)
		# Assuming passive mobs are not hostile, max health is less critical, default to 3.0
		self.passive_mobs = jax.lax.cond(
			jnp.logical_not(is_melee | is_ranged),
			lambda: _update_mob_array(self.passive_mobs, 3.0),
			lambda: self.passive_mobs,
		)

		self.mob_map = self.mob_map.at[level, position[0], position[1]].set(True)
		return self

	def add_mobs_randomly_near(
		self,
		rng: jax.Array,
		level: int,
		mob_name: str,
		type_id: int,
		n: int = 1,
		target_pos: jnp.ndarray = None,
		min_dist: int = 0,
		max_dist: int = 5,
		on_blocks: list[BlockType] = None,
	):
		"""Adds n mobs of a given type near a target on a specific level."""
		if target_pos is None:
			target_pos = self.player_position
		if on_blocks is None:
			on_blocks = [BlockType.GRASS, BlockType.PATH, BlockType.SAND]

		# This function's logic is nearly identical to add_mobs_randomly,
		# but with an added distance mask. We can reuse the same adaptation pattern.
		on_block_values = jnp.array([b.value for b in on_blocks])

		def _add_specific_mob_near(mob_array, max_mobs, mob_idx):
			num_active_mobs = mob_array.mask[level].sum()
			available_slots = max_mobs - num_active_mobs

			# Create distance mask
			rows_grid, cols_grid = jnp.indices(self.static_params.map_size)
			dist_map = jnp.abs(rows_grid - target_pos[0]) + jnp.abs(cols_grid - target_pos[1])
			distance_mask = (dist_map >= min_dist) & (dist_map <= max_dist)

			surface_mask = jnp.isin(self.map[level], on_block_values)
			valid_mask = distance_mask & surface_mask & jnp.logical_not(self.mob_map[level])
			num_valid_locations = valid_mask.sum()
			n_to_place = jnp.minimum(n, jnp.minimum(available_slots, num_valid_locations)).astype(
				jnp.int32
			)

			# The rest of the logic is identical to add_mobs_randomly
			random_scores = jax.random.uniform(rng, shape=self.static_params.map_size)
			masked_scores = jnp.where(valid_mask, random_scores, -jnp.inf)
			flat_indices = jnp.argsort(masked_scores.flatten())
			potential_indices = flat_indices[-max_mobs:]
			rows, cols = (
				potential_indices // self.static_params.map_size[1],
				potential_indices % self.static_params.map_size[1],
			)
			potential_positions = jnp.stack([rows, cols], axis=1)
			inactive_indices = jnp.where(
				jnp.logical_not(mob_array.mask[level]), size=max_mobs, fill_value=-1
			)[0]

			def loop_body(i, carry):
				current_mob_array, current_mob_map_slice = carry
				slot_to_fill = inactive_indices[i]
				position_to_set = potential_positions[i]
				updated_array = current_mob_array.replace(
					position=current_mob_array.position.at[level, slot_to_fill].set(
						position_to_set
					),
					health=current_mob_array.health.at[level, slot_to_fill].set(
						MOB_TYPE_HEALTH_MAPPING[type_id, mob_idx]
					),
					mask=current_mob_array.mask.at[level, slot_to_fill].set(True),
					type_id=current_mob_array.type_id.at[level, slot_to_fill].set(type_id),
				)
				updated_mob_map_slice = current_mob_map_slice.at[
					position_to_set[0], position_to_set[1]
				].set(True)
				return updated_array, updated_mob_map_slice

			final_mob_array, final_mob_map_slice = jax.lax.fori_loop(
				0, n_to_place, loop_body, (mob_array, self.mob_map[level])
			)
			return final_mob_array, final_mob_map_slice

		is_melee = mob_name == "melee"
		is_ranged = mob_name == "ranged"
		new_melee, melee_mob_map_slice = jax.lax.cond(
			is_melee,
			lambda: _add_specific_mob_near(
				self.melee_mobs, self.static_params.max_melee_mobs, MobType.MELEE.value
			),
			lambda: (self.melee_mobs, self.mob_map[level]),
		)
		new_ranged, ranged_mob_map_slice = jax.lax.cond(
			is_ranged,
			lambda: _add_specific_mob_near(
				self.ranged_mobs, self.static_params.max_ranged_mobs, MobType.RANGED.value
			),
			lambda: (self.ranged_mobs, self.mob_map[level]),
		)
		new_passive, passive_mob_map_slice = jax.lax.cond(
			jnp.logical_not(is_melee | is_ranged),
			lambda: _add_specific_mob_near(
				self.passive_mobs, self.static_params.max_passive_mobs, MobType.PASSIVE.value
			),
			lambda: (self.passive_mobs, self.mob_map[level]),
		)

		self.melee_mobs, self.ranged_mobs, self.passive_mobs = new_melee, new_ranged, new_passive
		self.mob_map = self.mob_map.at[level].set(
			melee_mob_map_slice | ranged_mob_map_slice | passive_mob_map_slice
		)

		return self

	def place_randomly(
		self,
		rng: jax.Array,
		level: int,
		block_type: BlockType,
		n: int = 1,
		on_blocks: list[BlockType] = None,
	):
		"""Places n blocks at random locations on a specific level."""
		if on_blocks is None:
			on_blocks = [BlockType.GRASS, BlockType.PATH, BlockType.FIRE_GRASS, BlockType.ICE_GRASS]
		on_block_values = jnp.array([b.value for b in on_blocks])

		# 1. Slice the specific level map we want to modify
		level_map = self.map[level]

		# 2. Call the pure helper function with the 2D map slice
		updated_level_map = _place_randomly_pure_jit_safe(
			rng, level_map, block_type.value, n, on_block_values
		)

		# 3. Update the slice in the main 3D map
		self.map = self.map.at[level].set(updated_level_map)
		return self

	def place_randomly_near(
		self,
		rng: jax.Array,
		level: int,
		block_type: BlockType,
		target_pos: tuple,
		min_dist: int,
		max_dist: int,
		n: int = 1,
		on_blocks: list[BlockType] = None,
	):
		"""Places n blocks near a target position on a specific level."""
		if on_blocks is None:
			on_blocks = [BlockType.GRASS, BlockType.PATH, BlockType.FIRE_GRASS, BlockType.ICE_GRASS]
		on_block_values = jnp.array([b.value for b in on_blocks])

		level_map = self.map[level]

		# --- Enforce minimum distance ---
		effective_min_dist = jnp.maximum(min_dist, 1)
		# Ensure max_dist is still >= effective_min_dist (optional but good practice)
		effective_max_dist = jnp.maximum(max_dist, effective_min_dist)
		# --- End enforcement ---

		updated_level_map = _place_randomly_near_pure(
			rng,
			level_map,
			self.static_params.map_size,
			block_type.value,
			n,
			target_pos,
			effective_min_dist,
			effective_max_dist,
			on_block_values,
		)

		self.map = self.map.at[level].set(updated_level_map)
		return self

	def place_adjacent_to_existing(
		self,
		rng: jax.Array,
		level: int,
		block_to_place: BlockType,
		target_block_type: BlockType,
		on_blocks: list[BlockType] = None,
	):
		"""Places a block adjacent to an existing block on a specific level."""
		if on_blocks is None:
			on_blocks = [BlockType.GRASS, BlockType.PATH, BlockType.FIRE_GRASS, BlockType.ICE_GRASS]
		on_block_values = jnp.array([b.value for b in on_blocks])

		level_map = self.map[level]
		map_height, map_width = self.static_params.map_size

		target_mask = level_map == target_block_type.value
		adjacent_mask = jnp.zeros_like(level_map, dtype=bool)

		# Convolve a 4-connectivity kernel with the target mask
		kernel = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
		adjacent_mask = jsp.signal.convolve(target_mask, kernel, mode="same") > 0

		surface_mask = jnp.isin(level_map, on_block_values)
		final_mask = adjacent_mask & surface_mask

		random_logits = jax.random.uniform(rng, shape=self.static_params.map_size)
		masked_logits = jnp.where(final_mask, random_logits, -jnp.inf)

		selected_flat_index = jnp.argmax(masked_logits.flatten())
		row, col = selected_flat_index // map_width, selected_flat_index % map_width

		is_valid = final_mask[row, col]
		new_value = jnp.where(is_valid, block_to_place.value, level_map[row, col])

		updated_level_map = level_map.at[row, col].set(new_value)
		self.map = self.map.at[level].set(updated_level_map)

		return self

	def build(self, rng: jax.Array) -> EnvState:
		"""Assembles the final, multi-level EnvState object from the builder's configuration."""
		rng, state_rng, potion_rng = jax.random.split(rng, 3)

		# Create empty projectile arrays, as they are not configured by the builder
		def _create_projectiles(max_num):
			projectiles = self.melee_mobs.replace(
				position=jnp.zeros((self.static_params.num_levels, max_num, 2), dtype=jnp.int32),
				health=jnp.ones((self.static_params.num_levels, max_num), dtype=jnp.float32),
				mask=jnp.zeros((self.static_params.num_levels, max_num), dtype=bool),
				attack_cooldown=jnp.zeros(
					(self.static_params.num_levels, max_num), dtype=jnp.int32
				),
				type_id=jnp.zeros((self.static_params.num_levels, max_num), dtype=jnp.int32),
			)
			directions = jnp.ones((self.static_params.num_levels, max_num, 2), dtype=jnp.int32)
			return projectiles, directions

		mob_projectiles, mob_projectile_directions = _create_projectiles(
			self.static_params.max_mob_projectiles
		)
		player_projectiles, player_projectile_directions = _create_projectiles(
			self.static_params.max_player_projectiles
		)

		state = EnvState(
			task_id=0,
			map=self.map,
			item_map=self.item_map,
			mob_map=self.mob_map,
			player_position=self.player_position,
			player_level=jnp.asarray(self.player_level, dtype=jnp.int32),
			player_direction=jnp.asarray(self.player_direction, dtype=jnp.int32),
			inventory=self.inventory,
			player_dexterity=jnp.asarray(self.player_dexterity, dtype=jnp.int32),
			player_strength=jnp.asarray(self.player_strength, dtype=jnp.int32),
			player_intelligence=jnp.asarray(self.player_intelligence, dtype=jnp.int32),
			sword_enchantment=jnp.asarray(self.sword_enchantment, dtype=jnp.int32),
			bow_enchantment=jnp.asarray(self.bow_enchantment, dtype=jnp.int32),
			armour_enchantments=self.armour_enchantments,
			learned_spells=self.learned_spells,
			melee_mobs=self.melee_mobs,
			ranged_mobs=self.ranged_mobs,
			passive_mobs=self.passive_mobs,
			growing_plants_positions=self.growing_plants_positions,
			growing_plants_age=self.growing_plants_age,
			growing_plants_mask=self.growing_plants_mask,
			monsters_killed=self.monsters_killed,
			player_health=jnp.asarray(9.0, dtype=jnp.float32),
			player_food=jnp.asarray(9, dtype=jnp.int32),
			player_drink=jnp.asarray(9, dtype=jnp.int32),
			player_energy=jnp.asarray(9, dtype=jnp.int32),
			player_mana=jnp.asarray(9, dtype=jnp.int32),
			player_xp=jnp.asarray(0, dtype=jnp.int32),
			is_sleeping=False,
			is_resting=False,
			player_recover=jnp.asarray(0.0, dtype=jnp.float32),
			player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
			player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
			player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
			player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
			achievements=jnp.zeros((len(Achievement),), dtype=bool),
			light_map=self.light_map,
			down_ladders=self.ladders_down,
			up_ladders=self.ladders_up,
			chests_opened=jnp.zeros(self.static_params.num_levels, dtype=bool),
			mob_projectiles=mob_projectiles,
			mob_projectile_directions=mob_projectile_directions,
			player_projectiles=player_projectiles,
			player_projectile_directions=player_projectile_directions,
			potion_mapping=jax.random.permutation(potion_rng, jnp.arange(6)),
			boss_progress=jnp.asarray(0, dtype=jnp.int32),
			boss_timesteps_to_spawn_this_round=jnp.asarray(50, dtype=jnp.int32),
			light_level=jnp.asarray(calculate_light_level(0, self.params), dtype=jnp.float32),
			timestep=jnp.asarray(0, dtype=jnp.int32),
			state_rng=state_rng,
			fractal_noise_angles=self.params.fractal_noise_angles,
			running_original_return=jnp.asarray(0.0, dtype=jnp.float32),
		)
		return state
