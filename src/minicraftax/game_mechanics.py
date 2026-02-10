import jax
import jax.numpy as jnp
from craftax.craftax.constants import (
	DIRECTIONS,
	FLOOR_MOB_MAPPING,
	FLOOR_MOB_SPAWN_CHANCE,
	LEVEL_ACHIEVEMENT_MAP,
	MOB_TYPE_COLLISION_MAPPING,
	MOB_TYPE_DAMAGE_MAPPING,
	MOB_TYPE_HEALTH_MAPPING,
	RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING,
	Achievement,
	Action,
	BlockType,
	ItemType,
	MobType,
	ProjectileType,
)
from craftax.craftax.craftax_state import Mobs
from craftax.craftax.util.game_logic_utils import (
	attack_mob,
	get_damage_done_to_player,
	get_max_energy,
	get_max_health,
	in_bounds,
	is_boss_spawn_wave,
	is_fighting_boss,
	is_in_mob,
	is_in_solid_block,
	is_position_in_bounds_not_in_mob_not_colliding,
	spawn_projectile,
)
from craftax.craftax.util.maths_utils import get_distance_map


def update_mobs(rng, state, params, static_params, task):
	"""Updates the state of all mobs in the environment.

	This involves moving mobs (melee, passive, ranged) and processing mob projectiles.
	Mob AI includes random movement, chasing the player, and attacking.

	Args:
		rng: JAX random number generator key.
		state: Current environment state.
		params: Dynamic environment parameters.
		static_params: Static environment parameters.
		task: Task-specific parameters and configuration.

	Returns:
		tuple: (rng, new_state) - The interaction returns a state but logic here expects returning state directly often.
		       Wait, the signature in code shows it returns `state`.
	"""
	# Move melee_mobs
	def _move_melee_mob(rng_and_state, melee_mob_index):
		rng, state = rng_and_state
		melee_mobs = state.melee_mobs

		# Random move
		rng, _rng = jax.random.split(rng)
		random_move_direction = jax.random.choice(
			_rng,
			DIRECTIONS[1:5],
		)
		random_move_proposed_position = (
			melee_mobs.position[state.player_level, melee_mob_index] + random_move_direction
		)

		# Move towards player
		player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
		player_move_direction_abs = jnp.abs(
			state.player_position - melee_mobs.position[state.player_level, melee_mob_index]
		)
		player_move_direction_index_p = (
			player_move_direction_abs == player_move_direction_abs.max()
		) / player_move_direction_abs.sum()
		rng, _rng = jax.random.split(rng)
		player_move_direction_index = jax.random.choice(
			_rng,
			jnp.arange(2),
			p=player_move_direction_index_p,
		)

		player_move_direction = player_move_direction.at[player_move_direction_index].set(
			jnp.sign(
				state.player_position[player_move_direction_index]
				- melee_mobs.position[state.player_level, melee_mob_index][
					player_move_direction_index
				]
			).astype(jnp.int32)
		)
		player_move_proposed_position = (
			melee_mobs.position[state.player_level, melee_mob_index] + player_move_direction
		)

		# Choose movement
		close_to_player = (
			jnp.sum(
				jnp.abs(
					melee_mobs.position[state.player_level, melee_mob_index] - state.player_position
				)
			)
			< task.melee_trigger_distance
		)
		close_to_player = jnp.logical_or(close_to_player, is_fighting_boss(state, static_params))

		rng, _rng = jax.random.split(rng)
		close_to_player = jnp.logical_and(close_to_player, jax.random.uniform(_rng) < 0.75)

		# Determine if the mob should chase the player or move randomly.
		# If within 'melee_trigger_distance' (e.g. 10 blocks) or fighting a boss, chase.
		proposed_position = jax.lax.select(
			close_to_player,
			player_move_proposed_position,
			random_move_proposed_position,
		)

		# Choose attack or not
		# If the mob is directly adjacent (Manhattan distance == 1) to the player,
		# its attack cooldown is 0 or less, and it's active (mask == True),
		# it decides to attack.
		is_attacking_player = (
			jnp.sum(
				jnp.abs(
					melee_mobs.position[state.player_level, melee_mob_index] - state.player_position
				)
			)
			== 1
		)
		is_attacking_player = jnp.logical_and(
			is_attacking_player,
			melee_mobs.attack_cooldown[state.player_level, melee_mob_index] <= 0,
		)
		is_attacking_player = jnp.logical_and(
			is_attacking_player, melee_mobs.mask[state.player_level, melee_mob_index]
		)

		proposed_position = jax.lax.select(
			is_attacking_player,
			melee_mobs.position[state.player_level, melee_mob_index],
			proposed_position,
		)

		melee_mob_base_damage = MOB_TYPE_DAMAGE_MAPPING[
			melee_mobs.type_id[state.player_level, melee_mob_index], MobType.MELEE.value
		]

		# Apply task-specific damage multiplier
		modified_melee_mob_base_damage = melee_mob_base_damage * task.mob_damage_multiplier
		melee_mob_damage = get_damage_done_to_player(
			# state, static_params, melee_mob_base_damage * (1 + 2.5 * state.is_sleeping)
			state,
			static_params,
			modified_melee_mob_base_damage * (1 + 2.5 * state.is_sleeping),  # Use modified damage
		)

		new_cooldown = jax.lax.select(
			is_attacking_player,
			5,  # Fixed cooldown of 5 steps after attacking
			melee_mobs.attack_cooldown[state.player_level, melee_mob_index] - 1,
		)

		is_waking_player = jnp.logical_and(state.is_sleeping, is_attacking_player)

		state = state.replace(
			player_health=state.player_health - melee_mob_damage * is_attacking_player,
			is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(is_attacking_player)),
			is_resting=jnp.logical_and(state.is_resting, jnp.logical_not(is_attacking_player)),
			achievements=state.achievements.at[Achievement.WAKE_UP.value].set(
				jnp.logical_or(state.achievements[Achievement.WAKE_UP.value], is_waking_player)
			),
		)

		mob_type = melee_mobs.type_id[state.player_level, melee_mob_index]
		collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 1]
		valid_move = is_position_in_bounds_not_in_mob_not_colliding(
			state, proposed_position, collision_map
		)
		position = jax.lax.select(
			valid_move,
			proposed_position,
			melee_mobs.position[state.player_level, melee_mob_index],
		)

		# checks if the mob is further than params.mob_despawn_distance from the player
		# if it's too far, the should_not_despawn flag becomes False
		# mobs on the boss level never despawn based on distance
		should_not_despawn = (
			jnp.abs(
				melee_mobs.position[state.player_level, melee_mob_index] - state.player_position
			).sum()
			< params.mob_despawn_distance  # POTENTIAL VARIABLE
		)
		should_not_despawn = jnp.logical_or(
			should_not_despawn, is_fighting_boss(state, static_params)
		)

		rng, _rng = jax.random.split(rng)

		# Clear our old entry if we are alive
		new_mob_map = state.mob_map.at[
			state.player_level,
			state.melee_mobs.position[state.player_level, melee_mob_index, 0],
			state.melee_mobs.position[state.player_level, melee_mob_index, 1],
		].set(
			jnp.logical_and(
				state.mob_map[
					state.player_level,
					state.melee_mobs.position[state.player_level, melee_mob_index, 0],
					state.melee_mobs.position[state.player_level, melee_mob_index, 1],
				],
				jnp.logical_not(melee_mobs.mask[state.player_level, melee_mob_index]),
			)
		)
		new_mask = jnp.logical_and(
			state.melee_mobs.mask[state.player_level, melee_mob_index],
			should_not_despawn,
		)
		# Enter new entry if we are alive and not despawning this timestep
		new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
			jnp.logical_or(new_mob_map[state.player_level, position[0], position[1]], new_mask)
		)

		state = state.replace(
			melee_mobs=state.melee_mobs.replace(
				position=state.melee_mobs.position.at[state.player_level, melee_mob_index].set(
					position
				),
				attack_cooldown=state.melee_mobs.attack_cooldown.at[
					state.player_level, melee_mob_index
				].set(new_cooldown),
				mask=state.melee_mobs.mask.at[state.player_level, melee_mob_index].set(new_mask),
			),
			mob_map=new_mob_map,
		)

		return (_rng, state), None

	rng, _rng = jax.random.split(rng)
	(rng, state), _ = jax.lax.scan(
		_move_melee_mob, (rng, state), jnp.arange(static_params.max_melee_mobs)
	)

	# Move passive_mobs
	def _move_passive_mob(rng_and_state, passive_mob_index):
		rng, state = rng_and_state
		passive_mobs = state.passive_mobs

		# Random move
		rng, _rng = jax.random.split(rng)
		random_move_direction = jax.random.choice(
			_rng,
			DIRECTIONS[1:9],  # 50% chance of not moving
		)
		proposed_position = (
			passive_mobs.position[state.player_level, passive_mob_index] + random_move_direction
		)

		mob_type = passive_mobs.type_id[state.player_level, passive_mob_index]
		collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 0]
		valid_move = is_position_in_bounds_not_in_mob_not_colliding(
			state, proposed_position, collision_map
		)
		position = jax.lax.select(
			valid_move,
			proposed_position,
			passive_mobs.position[state.player_level, passive_mob_index],
		)

		# same logic as with melee mobs for despawning
		should_not_despawn = (
			jnp.abs(
				passive_mobs.position[state.player_level, passive_mob_index] - state.player_position
			).sum()
			< params.mob_despawn_distance  # POTENTIAL VARIABLE
		)

		# Clear our old entry if we are alive
		new_mob_map = state.mob_map.at[
			state.player_level,
			state.passive_mobs.position[state.player_level, passive_mob_index, 0],
			state.passive_mobs.position[state.player_level, passive_mob_index, 1],
		].set(
			jnp.logical_and(
				state.mob_map[
					state.player_level,
					state.passive_mobs.position[state.player_level, passive_mob_index, 0],
					state.passive_mobs.position[state.player_level, passive_mob_index, 1],
				],
				jnp.logical_not(passive_mobs.mask[state.player_level, passive_mob_index]),
			)
		)
		new_mask = jnp.logical_and(
			state.passive_mobs.mask[state.player_level, passive_mob_index],
			should_not_despawn,
		)
		# Enter new entry if we are alive and not despawning this timestep
		new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
			jnp.logical_or(new_mob_map[state.player_level, position[0], position[1]], new_mask)
		)

		state = state.replace(
			passive_mobs=state.passive_mobs.replace(
				position=state.passive_mobs.position.at[state.player_level, passive_mob_index].set(
					position
				),
				mask=state.passive_mobs.mask.at[state.player_level, passive_mob_index].set(
					jnp.logical_and(
						state.passive_mobs.mask[state.player_level, passive_mob_index],
						should_not_despawn,
					)
				),
			),
			mob_map=new_mob_map,
		)

		return (rng, state), None

	rng, _rng = jax.random.split(rng)
	(rng, state), _ = jax.lax.scan(
		_move_passive_mob, (rng, state), jnp.arange(static_params.max_passive_mobs)
	)

	# Move ranged_mobs

	def _move_ranged_mob(rng_and_state, ranged_mob_index):
		rng, state = rng_and_state
		ranged_mobs = state.ranged_mobs

		# Random move
		rng, _rng = jax.random.split(rng)
		random_move_direction = jax.random.choice(
			_rng,
			DIRECTIONS[1:5],
		)
		random_move_proposed_position = (
			ranged_mobs.position[state.player_level, ranged_mob_index] + random_move_direction
		)

		# Move towards player
		player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
		player_move_direction_abs = jnp.abs(
			state.player_position - ranged_mobs.position[state.player_level, ranged_mob_index]
		)
		player_move_direction_index_p = (
			player_move_direction_abs == player_move_direction_abs.max()
		) / player_move_direction_abs.sum()
		rng, _rng = jax.random.split(rng)
		player_move_direction_index = jax.random.choice(
			_rng,
			jnp.arange(2),
			p=player_move_direction_index_p,
		)

		player_move_direction = player_move_direction.at[player_move_direction_index].set(
			jnp.sign(
				state.player_position[player_move_direction_index]
				- ranged_mobs.position[state.player_level, ranged_mob_index][
					player_move_direction_index
				]
			).astype(jnp.int32)
		)
		player_move_towards_proposed_position = (
			ranged_mobs.position[state.player_level, ranged_mob_index] + player_move_direction
		)
		player_move_away_proposed_position = (
			ranged_mobs.position[state.player_level, ranged_mob_index] - player_move_direction
		)

		# Choose movement
		distance_to_player = jnp.sum(
			jnp.abs(
				ranged_mobs.position[state.player_level, ranged_mob_index] - state.player_position
			)
		)

		far_from_player = distance_to_player >= 6
		too_close_to_player = distance_to_player <= 3

		proposed_position = jax.lax.select(
			far_from_player,
			player_move_towards_proposed_position,
			random_move_proposed_position,
		)
		proposed_position = jax.lax.select(
			too_close_to_player,
			player_move_away_proposed_position,
			proposed_position,
		)

		rng, _rng = jax.random.split(rng)

		# if distance >=6 calculated move is towards player
		# if distance <=3 calculated move is away from player
		# else calculated move is random
		# 15% chance to ignore calculated move and do random move instead
		proposed_position = jax.lax.select(
			jax.random.uniform(_rng) > 0.85,
			proposed_position,
			random_move_proposed_position,
		)

		# Choose attack or not
		# attacks if between 4 and 5 blocks away, <=3 and chosen step is blocked, cooldown is 0 or less, and active
		is_attacking_player = jnp.logical_and(distance_to_player >= 4, distance_to_player <= 5)
		# If we're too close to player (so we want to run) but are blocked, we shoot
		mob_type = ranged_mobs.type_id[state.player_level, ranged_mob_index]
		collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 2]
		is_attacking_player = jnp.logical_or(
			is_attacking_player,
			jnp.logical_and(
				too_close_to_player,
				jnp.logical_not(
					is_position_in_bounds_not_in_mob_not_colliding(
						state, proposed_position, collision_map
					)
				),
			),
		)

		is_attacking_player = jnp.logical_and(
			is_attacking_player,
			ranged_mobs.attack_cooldown[state.player_level, ranged_mob_index] <= 0,
		)
		is_attacking_player = jnp.logical_and(
			is_attacking_player, ranged_mobs.mask[state.player_level, ranged_mob_index]
		)

		# Spawn projectile
		can_spawn_projectile = (
			state.mob_projectiles.mask[state.player_level].sum() < static_params.max_mob_projectiles
		)
		new_projectile_position = ranged_mobs.position[state.player_level, ranged_mob_index]

		is_spawning_projectile = jnp.logical_and(is_attacking_player, can_spawn_projectile)

		new_mob_projectiles, new_mob_projectile_directions = spawn_projectile(
			state,
			static_params,
			state.mob_projectiles,
			state.mob_projectile_directions,
			new_projectile_position,
			is_spawning_projectile,
			player_move_direction,
			RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING[
				ranged_mobs.type_id[state.player_level, ranged_mob_index]
			],
		)

		state = state.replace(
			mob_projectiles=new_mob_projectiles,
			mob_projectile_directions=new_mob_projectile_directions,
		)

		proposed_position = jax.lax.select(
			is_attacking_player,
			ranged_mobs.position[state.player_level, ranged_mob_index],
			proposed_position,
		)

		new_cooldown = jax.lax.select(
			is_attacking_player,
			4,
			ranged_mobs.attack_cooldown[state.player_level, ranged_mob_index] - 1,
		)

		valid_move = is_position_in_bounds_not_in_mob_not_colliding(
			state, proposed_position, collision_map
		)

		position = jax.lax.select(
			valid_move,
			proposed_position,
			ranged_mobs.position[state.player_level, ranged_mob_index],
		)

		# same logic as with melee mobs for despawning
		should_not_despawn = (
			jnp.abs(
				ranged_mobs.position[state.player_level, ranged_mob_index] - state.player_position
			).sum()
			< params.mob_despawn_distance  # POTENTIAL VARIABLE
		)
		should_not_despawn = jnp.logical_or(
			should_not_despawn, is_fighting_boss(state, static_params)
		)

		# Clear our old entry if we are alive
		new_mob_map = state.mob_map.at[
			state.player_level,
			state.ranged_mobs.position[state.player_level, ranged_mob_index, 0],
			state.ranged_mobs.position[state.player_level, ranged_mob_index, 1],
		].set(
			jnp.logical_and(
				state.mob_map[
					state.player_level,
					state.ranged_mobs.position[state.player_level, ranged_mob_index, 0],
					state.ranged_mobs.position[state.player_level, ranged_mob_index, 1],
				],
				jnp.logical_not(ranged_mobs.mask[state.player_level, ranged_mob_index]),
			)
		)
		new_mask = jnp.logical_and(
			state.ranged_mobs.mask[state.player_level, ranged_mob_index],
			should_not_despawn,
		)
		# Enter new entry if we are alive and not despawning this timestep
		new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
			jnp.logical_or(new_mob_map[state.player_level, position[0], position[1]], new_mask)
		)

		state = state.replace(
			ranged_mobs=state.ranged_mobs.replace(
				position=state.ranged_mobs.position.at[state.player_level, ranged_mob_index].set(
					position
				),
				attack_cooldown=state.ranged_mobs.attack_cooldown.at[
					state.player_level, ranged_mob_index
				].set(new_cooldown),
				mask=state.ranged_mobs.mask.at[state.player_level, ranged_mob_index].set(
					jnp.logical_and(
						state.ranged_mobs.mask[state.player_level, ranged_mob_index],
						should_not_despawn,
					)
				),
			),
			mob_map=new_mob_map,
		)

		return (rng, state), None

	rng, _rng = jax.random.split(rng)
	(rng, state), _ = jax.lax.scan(
		_move_ranged_mob, (rng, state), jnp.arange(static_params.max_ranged_mobs)
	)

	# Move projectiles
	def _move_mob_projectile(rng_and_state, projectile_index):
		rng, state = rng_and_state
		projectiles = state.mob_projectiles

		proposed_position = (
			projectiles.position[state.player_level, projectile_index]
			+ state.mob_projectile_directions[state.player_level, projectile_index]
		)

		proposed_position_in_player = (proposed_position == state.player_position).all()

		proposed_position_in_bounds = in_bounds(state, proposed_position)
		in_wall = is_in_solid_block(state, proposed_position)
		in_wall = jnp.logical_and(
			in_wall,
			jnp.logical_not(
				state.map[state.player_level][proposed_position[0], proposed_position[1]]
				== BlockType.WATER.value
			),
		)  # Arrows can go over water
		in_mob = is_in_mob(state, proposed_position)

		continue_move = jnp.logical_and(proposed_position_in_bounds, jnp.logical_not(in_wall))
		continue_move = jnp.logical_and(continue_move, jnp.logical_not(in_mob))

		hit_player0 = jnp.logical_and(
			(
				projectiles.position[state.player_level, projectile_index] == state.player_position
			).all(),
			projectiles.mask[state.player_level, projectile_index],
		)

		hit_player1 = jnp.logical_and(
			proposed_position_in_player,
			projectiles.mask[state.player_level, projectile_index],
		)
		hit_player = jnp.logical_or(hit_player0, hit_player1)

		continue_move = jnp.logical_and(continue_move, jnp.logical_not(hit_player))

		position = proposed_position

		# Clear our old entry if we are alive
		new_mask = jnp.logical_and(
			continue_move, projectiles.mask[state.player_level, projectile_index]
		)

		hit_bench_or_furnace = jnp.logical_or(
			state.map[state.player_level, position[0], position[1]] == BlockType.FURNACE.value,
			state.map[state.player_level, position[0], position[1]]
			== BlockType.CRAFTING_TABLE.value,
		)
		removing_block = jnp.logical_and(
			hit_bench_or_furnace, projectiles.mask[state.player_level, projectile_index]
		)

		new_block = jax.lax.select(
			removing_block,
			BlockType.PATH.value,
			state.map[state.player_level, position[0], position[1]],
		)

		projectile_type = state.mob_projectiles.type_id[state.player_level, projectile_index]

		projectile_base_damage_vector = MOB_TYPE_DAMAGE_MAPPING[
			projectile_type, MobType.PROJECTILE.value
		]

		# Apply task-specific damage multiplier
		modified_projectile_damage_vector = (
			projectile_base_damage_vector * task.mob_damage_multiplier
		)
		projectile_damage = get_damage_done_to_player(
			state,
			static_params,
			# MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value],
			modified_projectile_damage_vector,  # Use modified damage
		)

		state = state.replace(
			mob_projectiles=state.mob_projectiles.replace(
				position=state.mob_projectiles.position.at[
					state.player_level, projectile_index
				].set(position),
				mask=state.mob_projectiles.mask.at[state.player_level, projectile_index].set(
					new_mask
				),
			),
			player_health=state.player_health - projectile_damage * hit_player,
			is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(hit_player)),
			is_resting=jnp.logical_and(state.is_resting, jnp.logical_not(hit_player)),
			map=state.map.at[state.player_level, position[0], position[1]].set(new_block),
		)

		return (rng, state), None

	rng, _rng = jax.random.split(rng)
	(rng, state), _ = jax.lax.scan(
		_move_mob_projectile,
		(rng, state),
		jnp.arange(static_params.max_mob_projectiles),
	)

	def _move_player_projectile(rng_and_state, projectile_index):
		rng, state = rng_and_state
		projectiles = state.player_projectiles

		projectile_type = state.player_projectiles.type_id[state.player_level, projectile_index]

		projectile_damage_vector = (
			MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value]
			* projectiles.mask[state.player_level, projectile_index]
		)

		is_arrow = jnp.logical_or(
			projectile_type == ProjectileType.ARROW.value,
			projectile_type == ProjectileType.ARROW2.value,
		)

		# Bow enchantment
		arrow_damage_add = jnp.zeros(3, dtype=jnp.float32)
		arrow_damage_add = arrow_damage_add.at[state.bow_enchantment].set(
			projectile_damage_vector[0] / 2
		)
		arrow_damage_add = arrow_damage_add.at[0].set(0)

		projectile_damage_vector += jax.lax.select(
			is_arrow,
			arrow_damage_add,
			jnp.zeros(3, dtype=jnp.float32),
		)

		# Apply attribute scaling
		arrow_damage_coeff = 1 + 0.2 * (state.player_dexterity - 1)
		magic_damage_coeff = 1 + 0.5 * (state.player_intelligence - 1)

		projectile_damage_vector *= jax.lax.select(
			is_arrow,
			arrow_damage_coeff,
			1.0,
		)

		projectile_damage_vector *= jax.lax.select(
			jnp.logical_or(
				projectile_type == ProjectileType.FIREBALL.value,
				projectile_type == ProjectileType.ICEBALL.value,
			),
			magic_damage_coeff,
			1.0,
		)

		proposed_position = (
			projectiles.position[state.player_level, projectile_index]
			+ state.player_projectile_directions[state.player_level, projectile_index]
		)

		proposed_position_in_bounds = in_bounds(state, proposed_position)
		in_wall = is_in_solid_block(state, proposed_position)
		in_wall = jnp.logical_and(
			in_wall,
			jnp.logical_not(
				state.map[state.player_level][proposed_position[0], proposed_position[1]]
				== BlockType.WATER.value
			),
		)  # Arrows can go over water

		state, did_attack_mob0, did_kill_mob0 = attack_mob(
			state,
			projectiles.position[state.player_level, projectile_index],
			projectile_damage_vector,
			False,
		)

		projectile_damage_vector = projectile_damage_vector * (1 - did_attack_mob0)

		state, did_attack_mob1, did_kill_mob1 = attack_mob(
			state, proposed_position, projectile_damage_vector, False
		)

		did_attack_mob = jnp.logical_or(did_attack_mob0, did_attack_mob1)

		continue_move = jnp.logical_and(proposed_position_in_bounds, jnp.logical_not(in_wall))
		continue_move = jnp.logical_and(continue_move, jnp.logical_not(did_attack_mob))
		position = proposed_position

		# Clear our old entry if we are alive
		new_mask = jnp.logical_and(
			continue_move, projectiles.mask[state.player_level, projectile_index]
		)

		state = state.replace(
			player_projectiles=state.player_projectiles.replace(
				position=state.player_projectiles.position.at[
					state.player_level, projectile_index
				].set(position),
				mask=state.player_projectiles.mask.at[state.player_level, projectile_index].set(
					new_mask
				),
			),
		)

		return (rng, state), None

	rng, _rng = jax.random.split(rng)
	(rng, state), _ = jax.lax.scan(
		_move_player_projectile,
		(rng, state),
		jnp.arange(static_params.max_player_projectiles),
	)

	return state


def spawn_mobs(state, rng, params, static_params, task):
	"""Spawns new mobs (passive, melee, ranged) based on game rules and randomness.

	Args:
		state: Current environment state.
		rng: JAX random key.
		params: Dynamic parameters.
		static_params: Static parameters.
		task: Task-specific parameters.

	Returns:
		EnvState: Updated state with newly spawned mobs.
	"""
	player_distance_map = get_distance_map(state.player_position, static_params.map_size)
	grave_map = jnp.logical_or(
		state.map[state.player_level] == BlockType.GRAVE.value,
		jnp.logical_or(
			state.map[state.player_level] == BlockType.GRAVE2.value,
			state.map[state.player_level] == BlockType.GRAVE3.value,
		),
	)

	monster_spawn_coeff = (
		1
		+ (state.monsters_killed[state.player_level] < task.monsters_killed_to_clear_level)
		* 2
	)  # Triple spawn rate if we are on an uncleared level

	monster_spawn_coeff *= jax.lax.select(
		is_fighting_boss(state, static_params),
		is_boss_spawn_wave(state, static_params) * 1000,
		1,
	)

	# Passive mobs
	can_spawn_passive_mob = (
		state.passive_mobs.mask[state.player_level].sum() < static_params.max_passive_mobs
	)

	rng, _rng = jax.random.split(rng)
	passive_spawn_chance = (
		FLOOR_MOB_SPAWN_CHANCE[state.player_level, 0] * task.passive_spawn_multiplier
	)
	can_spawn_passive_mob = jnp.logical_and(
		can_spawn_passive_mob,
		# jax.random.uniform(_rng) < FLOOR_MOB_SPAWN_CHANCE[state.player_level, 0],
		jax.random.uniform(_rng) < passive_spawn_chance,
	)

	can_spawn_passive_mob = jnp.logical_and(
		can_spawn_passive_mob, jnp.logical_not(is_fighting_boss(state, static_params))
	)

	all_valid_blocks_map = jnp.logical_or(
		state.map[state.player_level] == BlockType.GRASS.value,
		jnp.logical_or(
			state.map[state.player_level] == BlockType.PATH.value,
			jnp.logical_or(
				state.map[state.player_level] == BlockType.FIRE_GRASS.value,
				state.map[state.player_level] == BlockType.ICE_GRASS.value,
			),
		),
	)
	grass_map = state.map[state.player_level] == BlockType.GRASS.value
	path_map = state.map[state.player_level] == BlockType.PATH.value
	new_passive_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.PASSIVE.value]

	passive_mobs_can_spawn_map = all_valid_blocks_map

	passive_mobs_can_spawn_map = jnp.logical_and(
		passive_mobs_can_spawn_map, player_distance_map > 3
	)
	passive_mobs_can_spawn_map = jnp.logical_and(
		passive_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
	)
	passive_mobs_can_spawn_map = jnp.logical_and(
		passive_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
	)
	can_spawn_passive_mob = jnp.logical_and(
		can_spawn_passive_mob, passive_mobs_can_spawn_map.sum() > 0
	)

	rng, _rng = jax.random.split(rng)
	passive_mob_position = jax.random.choice(
		_rng,
		jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
		shape=(1,),
		p=jnp.reshape(passive_mobs_can_spawn_map, -1) / jnp.sum(passive_mobs_can_spawn_map),
	)
	passive_mob_position = jnp.array(
		[
			passive_mob_position // static_params.map_size[0],
			passive_mob_position % static_params.map_size[1],
		]
	).T.astype(jnp.int32)[0]

	new_passive_mob_index = jnp.argmax(jnp.logical_not(state.passive_mobs.mask[state.player_level]))

	new_passive_mob_position = jax.lax.select(
		can_spawn_passive_mob,
		passive_mob_position,
		state.passive_mobs.position[state.player_level, new_passive_mob_index],
	)

	# Apply health multiplier
	base_passive_health = MOB_TYPE_HEALTH_MAPPING[new_passive_mob_type, MobType.PASSIVE.value]
	modified_passive_health = base_passive_health * task.mob_health_multiplier
	new_passive_mob_health = jax.lax.select(
		can_spawn_passive_mob,
		# MOB_TYPE_HEALTH_MAPPING[new_passive_mob_type, MobType.PASSIVE.value],
		modified_passive_health,  # Use modified health
		state.passive_mobs.health[state.player_level, new_passive_mob_index],
	)

	new_passive_mob_mask = jax.lax.select(
		can_spawn_passive_mob,
		True,
		state.passive_mobs.mask[state.player_level, new_passive_mob_index],
	)

	passive_mobs = Mobs(
		position=state.passive_mobs.position.at[state.player_level, new_passive_mob_index].set(
			new_passive_mob_position
		),
		health=state.passive_mobs.health.at[state.player_level, new_passive_mob_index].set(
			new_passive_mob_health
		),
		mask=state.passive_mobs.mask.at[state.player_level, new_passive_mob_index].set(
			new_passive_mob_mask
		),
		attack_cooldown=state.passive_mobs.attack_cooldown,
		type_id=state.passive_mobs.type_id.at[state.player_level, new_passive_mob_index].set(
			new_passive_mob_type
		),
	)

	state = state.replace(
		passive_mobs=passive_mobs,
		mob_map=state.mob_map.at[
			state.player_level, new_passive_mob_position[0], new_passive_mob_position[1]
		].set(
			jnp.logical_or(
				state.mob_map[
					state.player_level,
					new_passive_mob_position[0],
					new_passive_mob_position[1],
				],
				new_passive_mob_mask,
			)
		),
	)

	# Monsters
	monsters_can_spawn_player_range_map = player_distance_map > 9
	monsters_can_spawn_player_range_map_boss = player_distance_map <= 6

	monsters_can_spawn_player_range_map = jax.lax.select(
		is_fighting_boss(state, static_params),
		monsters_can_spawn_player_range_map_boss,
		monsters_can_spawn_player_range_map,
	)

	# Melee mobs
	can_spawn_melee_mob = (
		state.melee_mobs.mask[state.player_level].sum() < static_params.max_melee_mobs
	)

	new_melee_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.MELEE.value]
	new_melee_mob_type_boss = FLOOR_MOB_MAPPING[state.boss_progress, MobType.MELEE.value]

	new_melee_mob_type = jax.lax.select(
		is_fighting_boss(state, static_params),
		new_melee_mob_type_boss,
		new_melee_mob_type,
	)

	rng, _rng = jax.random.split(rng)
	# Calculate base chance with multiplier
	melee_base_chance = FLOOR_MOB_SPAWN_CHANCE[state.player_level, 1]
	# Calculate night bonus with multiplier
	melee_night_bonus = FLOOR_MOB_SPAWN_CHANCE[state.player_level, 3] * jnp.square(
		1 - state.light_level
	)
	# melee_mob_spawn_chance = FLOOR_MOB_SPAWN_CHANCE[
	#     state.player_level, 1
	# ] + FLOOR_MOB_SPAWN_CHANCE[state.player_level, 3] * jnp.square(
	#     1 - state.light_level
	# )
	melee_mob_spawn_chance = (melee_base_chance + melee_night_bonus) * task.melee_spawn_multiplier
	can_spawn_melee_mob = jnp.logical_and(
		can_spawn_melee_mob,
		jax.random.uniform(_rng) < melee_mob_spawn_chance * monster_spawn_coeff,
	)

	melee_mobs_can_spawn_map = jax.lax.select(
		is_fighting_boss(state, static_params), grave_map, all_valid_blocks_map
	)

	melee_mobs_can_spawn_map = jnp.logical_and(
		melee_mobs_can_spawn_map, monsters_can_spawn_player_range_map
	)
	melee_mobs_can_spawn_map = jnp.logical_and(
		melee_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
	)
	melee_mobs_can_spawn_map = jnp.logical_and(
		melee_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
	)

	can_spawn_melee_mob = jnp.logical_and(can_spawn_melee_mob, melee_mobs_can_spawn_map.sum() > 0)

	rng, _rng = jax.random.split(rng)
	melee_mob_position = jax.random.choice(
		_rng,
		jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
		shape=(1,),
		p=jnp.reshape(melee_mobs_can_spawn_map, -1) / jnp.sum(melee_mobs_can_spawn_map),
	)
	melee_mob_position = jnp.array(
		[
			melee_mob_position // static_params.map_size[0],
			melee_mob_position % static_params.map_size[1],
		]
	).T.astype(jnp.int32)[0]

	new_melee_mob_index = jnp.argmax(jnp.logical_not(state.melee_mobs.mask[state.player_level]))

	new_melee_mob_position = jax.lax.select(
		can_spawn_melee_mob,
		melee_mob_position,
		state.melee_mobs.position[state.player_level, new_melee_mob_index],
	)

	# Apply health multiplier
	base_melee_health = MOB_TYPE_HEALTH_MAPPING[new_melee_mob_type, MobType.MELEE.value]
	modified_melee_health = base_melee_health * task.mob_health_multiplier
	new_melee_mob_health = jax.lax.select(
		can_spawn_melee_mob,
		# MOB_TYPE_HEALTH_MAPPING[new_melee_mob_type, MobType.MELEE.value],
		modified_melee_health,  # Use modified health
		state.melee_mobs.health[state.player_level, new_melee_mob_index],
	)

	new_melee_mob_mask = jax.lax.select(
		can_spawn_melee_mob,
		True,
		state.melee_mobs.mask[state.player_level, new_melee_mob_index],
	)

	melee_mobs = Mobs(
		position=state.melee_mobs.position.at[state.player_level, new_melee_mob_index].set(
			new_melee_mob_position
		),
		health=state.melee_mobs.health.at[state.player_level, new_melee_mob_index].set(
			new_melee_mob_health
		),
		mask=state.melee_mobs.mask.at[state.player_level, new_melee_mob_index].set(
			new_melee_mob_mask
		),
		attack_cooldown=state.melee_mobs.attack_cooldown,
		type_id=state.melee_mobs.type_id.at[state.player_level, new_melee_mob_index].set(
			new_melee_mob_type
		),
	)

	state = state.replace(
		melee_mobs=melee_mobs,
		mob_map=state.mob_map.at[
			state.player_level, new_melee_mob_position[0], new_melee_mob_position[1]
		].set(
			jnp.logical_or(
				state.mob_map[
					state.player_level,
					new_melee_mob_position[0],
					new_melee_mob_position[1],
				],
				new_melee_mob_mask,
			)
		),
	)

	# Ranged mobs
	can_spawn_ranged_mob = (
		state.ranged_mobs.mask[state.player_level].sum() < static_params.max_ranged_mobs
	)

	new_ranged_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.RANGED.value]
	new_ranged_mob_type_boss = FLOOR_MOB_MAPPING[state.boss_progress, MobType.RANGED.value]

	new_ranged_mob_type = jax.lax.select(
		is_fighting_boss(state, static_params),
		new_ranged_mob_type_boss,
		new_ranged_mob_type,
	)

	rng, _rng = jax.random.split(rng)
	ranged_spawn_chance = (
		FLOOR_MOB_SPAWN_CHANCE[state.player_level, 2] * task.ranged_spawn_multiplier
	)
	can_spawn_ranged_mob = jnp.logical_and(
		can_spawn_ranged_mob,
		jax.random.uniform(_rng) < ranged_spawn_chance * monster_spawn_coeff,
	)

	# Hack for deep thing
	ranged_mobs_can_spawn_map = jax.lax.select(
		new_ranged_mob_type == 5,
		state.map[state.player_level] == BlockType.WATER.value,
		all_valid_blocks_map,
	)
	ranged_mobs_can_spawn_map = jax.lax.select(
		is_fighting_boss(state, static_params), grave_map, ranged_mobs_can_spawn_map
	)

	ranged_mobs_can_spawn_map = jnp.logical_and(
		ranged_mobs_can_spawn_map, monsters_can_spawn_player_range_map
	)
	ranged_mobs_can_spawn_map = jnp.logical_and(
		ranged_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
	)
	ranged_mobs_can_spawn_map = jnp.logical_and(
		ranged_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
	)

	can_spawn_ranged_mob = jnp.logical_and(
		can_spawn_ranged_mob, ranged_mobs_can_spawn_map.sum() > 0
	)

	rng, _rng = jax.random.split(rng)
	ranged_mob_position = jax.random.choice(
		_rng,
		jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
		shape=(1,),
		p=jnp.reshape(ranged_mobs_can_spawn_map, -1) / jnp.sum(ranged_mobs_can_spawn_map),
	)
	ranged_mob_position = jnp.array(
		[
			ranged_mob_position // static_params.map_size[0],
			ranged_mob_position % static_params.map_size[1],
		]
	).T.astype(jnp.int32)[0]

	new_ranged_mob_index = jnp.argmax(jnp.logical_not(state.ranged_mobs.mask[state.player_level]))

	new_ranged_mob_position = jax.lax.select(
		can_spawn_ranged_mob,
		ranged_mob_position,
		state.ranged_mobs.position[state.player_level, new_ranged_mob_index],
	)

	# Apply health multiplier
	base_ranged_health = MOB_TYPE_HEALTH_MAPPING[new_ranged_mob_type, MobType.RANGED.value]
	modified_ranged_health = base_ranged_health * task.mob_health_multiplier
	new_ranged_mob_health = jax.lax.select(
		can_spawn_ranged_mob,
		# MOB_TYPE_HEALTH_MAPPING[new_ranged_mob_type, MobType.RANGED.value],
		modified_ranged_health,  # Use modified health
		state.ranged_mobs.health[state.player_level, new_ranged_mob_index],
	)

	new_ranged_mob_mask = jax.lax.select(
		can_spawn_ranged_mob,
		True,
		state.ranged_mobs.mask[state.player_level, new_ranged_mob_index],
	)

	ranged_mobs = Mobs(
		position=state.ranged_mobs.position.at[state.player_level, new_ranged_mob_index].set(
			new_ranged_mob_position
		),
		health=state.ranged_mobs.health.at[state.player_level, new_ranged_mob_index].set(
			new_ranged_mob_health
		),
		mask=state.ranged_mobs.mask.at[state.player_level, new_ranged_mob_index].set(
			new_ranged_mob_mask
		),
		attack_cooldown=state.ranged_mobs.attack_cooldown,
		type_id=state.ranged_mobs.type_id.at[state.player_level, new_ranged_mob_index].set(
			new_ranged_mob_type
		),
	)

	state = state.replace(
		ranged_mobs=ranged_mobs,
		mob_map=state.mob_map.at[
			state.player_level, new_ranged_mob_position[0], new_ranged_mob_position[1]
		].set(
			jnp.logical_or(
				state.mob_map[
					state.player_level,
					new_ranged_mob_position[0],
					new_ranged_mob_position[1],
				],
				new_ranged_mob_mask,
			)
		),
	)

	return state


def change_floor(state, action, env_params, static_params, task):
	on_down_ladder = (
		state.item_map[state.player_level, state.player_position[0], state.player_position[1]]
		== ItemType.LADDER_DOWN.value
	)
	is_moving_down = jnp.logical_and(
		action == Action.DESCEND.value,
		jnp.logical_or(
			env_params.god_mode,
			jnp.logical_and(
				on_down_ladder,
				state.monsters_killed[state.player_level] >= task.monsters_killed_to_clear_level,
			),
		),
	)
	is_moving_down = jnp.logical_and(
		is_moving_down, state.player_level < static_params.num_levels - 1
	)

	moving_down_position = state.up_ladders[state.player_level + 1]

	on_up_ladder = (
		state.item_map[state.player_level, state.player_position[0], state.player_position[1]]
		== ItemType.LADDER_UP.value
	)
	is_moving_up = jnp.logical_and(
		action == Action.ASCEND.value,
		jnp.logical_or(
			env_params.god_mode,
			on_up_ladder,
		),
	)
	is_moving_up = jnp.logical_and(is_moving_up, state.player_level > 0)

	moving_up_position = state.down_ladders[state.player_level - 1]

	is_not_moving = jnp.logical_not(jnp.logical_or(is_moving_up, is_moving_down))

	delta_floor = 1 * is_moving_down - 1 * is_moving_up
	position = (
		(state.player_position * is_not_moving)
		+ (is_moving_down * moving_down_position)
		+ (is_moving_up * moving_up_position)
	)

	move_down_achievement = LEVEL_ACHIEVEMENT_MAP[state.player_level + delta_floor]

	new_achievements = state.achievements.at[move_down_achievement].set(
		jnp.logical_or(
			(state.player_level + delta_floor) != 0,
			state.achievements[move_down_achievement],
		)
	)

	new_floor = jnp.logical_and(
		(state.player_level + delta_floor) != 0,
		jnp.logical_not(state.achievements[move_down_achievement]),
	)

	state = state.replace(
		player_level=state.player_level + delta_floor,
		player_position=position,
		achievements=new_achievements,
		player_xp=state.player_xp + 1 * new_floor,
	)

	return state


def update_player_intrinsics(state, action, static_params, task):
	# Start sleeping?
	is_starting_sleep = jnp.logical_and(
		action == Action.SLEEP.value, state.player_energy < get_max_energy(state)
	)
	new_is_sleeping = jnp.logical_or(state.is_sleeping, is_starting_sleep)
	state = state.replace(is_sleeping=new_is_sleeping)

	# Wake up?
	is_waking_up = jnp.logical_and(state.player_energy >= get_max_energy(state), state.is_sleeping)
	new_is_sleeping = jnp.logical_and(state.is_sleeping, jnp.logical_not(is_waking_up))
	state = state.replace(
		is_sleeping=new_is_sleeping,
		achievements=state.achievements.at[Achievement.WAKE_UP.value].set(
			jnp.logical_or(state.achievements[Achievement.WAKE_UP.value], is_waking_up)
		),
	)

	# Start resting?
	is_starting_rest = jnp.logical_and(
		action == Action.REST.value, state.player_health < get_max_health(state)
	)
	new_is_resting = jnp.logical_or(state.is_resting, is_starting_rest)
	state = state.replace(is_resting=new_is_resting)

	# Wake up from resting
	is_waking_up = jnp.logical_and(
		state.is_resting,
		jnp.logical_or(
			state.player_health >= get_max_health(state),
			jnp.logical_or(state.player_food <= 0, state.player_drink <= 0),
		),
	)
	new_is_resting = jnp.logical_and(state.is_resting, jnp.logical_not(is_waking_up))
	state = state.replace(
		is_resting=new_is_resting,
	)

	not_boss = jnp.logical_not(is_fighting_boss(state, static_params))

	intrinsic_decay_coeff = 1.0 - (0.125 * (state.player_dexterity - 1))

	# Hunger
	hunger_add = jax.lax.select(state.is_sleeping, 0.5, 1.0) * intrinsic_decay_coeff
	hunger_add *= task.needs_depletion_multiplier
	new_hunger = state.player_hunger + hunger_add

	hungered_food = jnp.maximum(state.player_food - 1 * not_boss, 0)
	new_food = jax.lax.select(new_hunger > 25, hungered_food, state.player_food)
	new_hunger = jax.lax.select(new_hunger > 25, 0.0, new_hunger)

	state = state.replace(
		player_hunger=new_hunger,
		player_food=new_food,
	)

	# Thirst
	thirst_add = jax.lax.select(state.is_sleeping, 0.5, 1.0) * intrinsic_decay_coeff
	thirst_add *= task.needs_depletion_multiplier
	new_thirst = state.player_thirst + thirst_add
	thirsted_drink = jnp.maximum(state.player_drink - 1 * not_boss, 0)
	new_drink = jax.lax.select(new_thirst > 20, thirsted_drink, state.player_drink)
	new_thirst = jax.lax.select(new_thirst > 20, 0.0, new_thirst)

	state = state.replace(
		player_thirst=new_thirst,
		player_drink=new_drink,
	)

	# Fatigue
	fatige_increase_rate = intrinsic_decay_coeff * task.needs_depletion_multiplier
	new_fatigue = jax.lax.select(
		state.is_sleeping,
		jnp.minimum(state.player_fatigue - 1, 0),
		state.player_fatigue + fatige_increase_rate,
	)

	new_energy = jax.lax.select(
		new_fatigue > 30,
		jnp.maximum(state.player_energy - 1 * not_boss, 0),
		state.player_energy,
	)
	new_fatigue = jax.lax.select(new_fatigue > 30, 0.0, new_fatigue)

	new_energy = jax.lax.select(
		new_fatigue < -10,
		jnp.minimum(state.player_energy + 1, get_max_energy(state)),
		new_energy,
	)
	new_fatigue = jax.lax.select(new_fatigue < -10, 0.0, new_fatigue)

	state = state.replace(
		player_fatigue=new_fatigue,
		player_energy=new_energy,
	)

	# Health
	necessities = jnp.array(
		[
			state.player_food > 0,
			state.player_drink > 0,
			jnp.logical_or(state.player_energy > 0, state.is_sleeping),
		],
		dtype=bool,
	)

	all_necessities = necessities.all()
	recover_all = jax.lax.select(state.is_sleeping, 2.0, 1.0) * task.health_recover_multiplier
	recover_not_all = (
		jax.lax.select(state.is_sleeping, -0.5, -1.0) * not_boss * task.health_loss_multiplier
	)
	recover_add = jax.lax.select(all_necessities, recover_all, recover_not_all)

	new_recover = state.player_recover + recover_add

	recovered_health = jnp.minimum(state.player_health + 1, get_max_health(state))
	derecovered_health = state.player_health - 1

	new_health = jax.lax.select(new_recover > 25, recovered_health, state.player_health)
	new_recover = jax.lax.select(new_recover > 25, 0.0, new_recover)
	new_health = jax.lax.select(new_recover < -15, derecovered_health, new_health)
	new_recover = jax.lax.select(new_recover < -15, 0.0, new_recover)

	state = state.replace(
		player_recover=new_recover,
		player_health=new_health,
	)

	# Mana
	mana_recover_coeff = 1 + 0.25 * (state.player_intelligence - 1) * task.mana_recover_multiplier
	new_recover_mana = (
		jax.lax.select(
			state.is_sleeping,
			state.player_recover_mana + 2,
			state.player_recover_mana + 1,
		)
		* mana_recover_coeff
	)

	new_mana = jax.lax.select(new_recover_mana > 30, state.player_mana + 1, state.player_mana)
	new_recover_mana = jax.lax.select(new_recover_mana > 30, 0.0, new_recover_mana)

	state = state.replace(
		player_recover_mana=new_recover_mana,
		player_mana=new_mana,
	)

	return state


def update_plants(state, static_params, task):
	growing_plants_age = state.growing_plants_age + 1
	growing_plants_age *= state.growing_plants_mask

	finished_growing_plants = growing_plants_age >= task.growing_plants_age

	new_plant_blocks = jnp.where(
		finished_growing_plants,
		BlockType.RIPE_PLANT.value,
		BlockType.PLANT.value,
	)

	def _set_plant_block(map, plant_index):
		new_block = jax.lax.select(
			finished_growing_plants[plant_index],
			new_plant_blocks[plant_index],
			map[
				state.growing_plants_positions[plant_index][0],
				state.growing_plants_positions[plant_index][1],
			],
		)
		map = map.at[
			state.growing_plants_positions[plant_index][0],
			state.growing_plants_positions[plant_index][1],
		].set(new_block)
		return map, None

	new_map, _ = jax.lax.scan(
		_set_plant_block,
		state.map[0],
		jnp.arange(static_params.max_growing_plants),
	)

	new_whole_map = state.map.at[0].set(new_map)

	state = state.replace(
		map=new_whole_map,
		growing_plants_age=growing_plants_age,
	)

	return state
