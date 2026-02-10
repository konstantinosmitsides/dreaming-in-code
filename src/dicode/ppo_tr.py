import time

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from dicode.network import ActorCriticTransformer, Transition

from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv, MultiTaskMiniCraftaxEnvR
from dicode.wrappers_cl import (
	DistributedMultiTaskOptimisticLogWrapper,
	MultiTaskOptimisticLogWrapperAllTasks,
)


# --- 2. Transformer Network Class ---
# --- 2. Transformer Network Class ---
# Imported from dicode.network


# --- Helper Functions for Transformer Logic ---
indices_select = lambda x, y: x[y]
batch_indices_select = jax.vmap(indices_select)
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


def make_train(
	config,
	task_classes,
	num_training_updates,
	task_embeddings=None,
	task_distribution_proportions=None,
	initial_global_update_step=0,
):
	"""Sets up the environment, network, and returns the JIT-compiled train function."""
	# --- Environment Setup (IDENTICAL TO OLD CODE) ---
	NUM_UPDATES = num_training_updates
	num_tasks = len(task_classes)

	static_env_params = StaticEnvParams()
	env_params = EnvParams(max_timesteps=4096)

	if config.mode != "reward":
		if task_embeddings is not None:
			embedding_size = task_embeddings.shape[1]
			print(f"Using embedding size: {embedding_size}")
			base_env = MultiTaskMiniCraftaxEnv(
				task_classes,
				static_env_params,
				env_params,
				config.condition_on_task,
				conditioning_type="embedding",
				embedding_size=embedding_size,
				completion_bonus_scale=config.completion_bonus_scale,
				completion_bonus_min=config.completion_bonus_min,
				bonus_type=config.bonus_type,
				dynamic_bonus_k=config.dynamic_bonus_k,
			)
		else:
			base_env = MultiTaskMiniCraftaxEnv(
				task_classes,
				static_env_params,
				env_params,
				config.condition_on_task,
				completion_bonus_scale=config.completion_bonus_scale,
				completion_bonus_min=config.completion_bonus_min,
				bonus_type=config.bonus_type,
				dynamic_bonus_k=config.dynamic_bonus_k,
			)
	else:
		if task_embeddings is not None:
			embedding_size = task_embeddings.shape[1]
			base_env = MultiTaskMiniCraftaxEnvR(
				task_classes,
				static_env_params,
				env_params,
				config.condition_on_task,
				conditioning_type="embedding",
				embedding_size=embedding_size,
				completion_bonus_scale=config.completion_bonus_scale,
				completion_bonus_min=config.completion_bonus_min,
			)
		else:
			base_env = MultiTaskMiniCraftaxEnvR(
				task_classes,
				static_env_params,
				env_params,
				config.condition_on_task,
				completion_bonus_scale=config.completion_bonus_scale,
				completion_bonus_min=config.completion_bonus_min,
			)

	if task_distribution_proportions is None:
		# Default to uniform distribution if not provided
		task_distribution_proportions = jnp.ones(num_tasks) / num_tasks

	env = DistributedMultiTaskOptimisticLogWrapper(
		base_env,
		jax.random.PRNGKey(0),  # We need a key for the permutation in the wrapper
		config.num_envs,
		num_tasks,
		config.optimistic_reset_ratio,
		task_distribution_proportions,
		task_embeddings,
	)
	env_params = env.default_params

	# --- Network Setup (CHANGED TO TRANSFORMER) ---
	# NOTE: You must ensure your config has these keys (embed_size, num_heads, etc.)
	# We map existing config keys where possible, or expect new ones.
	network = ActorCriticTransformer(
		action_dim=env.action_space(env_params).n,
		activation=config.activation,
		hidden_layers=config.hidden_layers,  # Mapping layer_size to hidden_layers
		encoder_size=config.embed_size,  # Mapping embedding_size to encoder_size/embed_size
		num_heads=config.num_heads,
		qkv_features=config.qkv_features,
		num_layers=config.num_layers,
		gating=config.gating,
		gating_bias=config.gating_bias,
	)

	# --- Optimizer Setup ---
	TOTAL_GLOBAL_UPDATES = (
		(
			config.total_timesteps
			// config.num_envs
			// config.num_steps
			// config.max_updates_per_session
		)
		+ 1
	) * config.max_updates_per_session

	def linear_schedule(count):
		frac = (
			1.0 - (count // (config.num_minibatches * config.update_epochs)) / TOTAL_GLOBAL_UPDATES
		)
		return config.min_lr + (config.lr - config.min_lr) * frac

	if config.anneal_lr:
		tx = optax.chain(
			optax.clip_by_global_norm(config.max_grad_norm),
			optax.adam(learning_rate=linear_schedule, eps=1e-5),
		)
	else:
		tx = optax.chain(
			optax.clip_by_global_norm(config.max_grad_norm),
			optax.adam(config.lr, eps=1e-5),
		)

	def train(rng, train_state=None, current_original_return=0.0):
		"""The core JIT-compiled function."""
		obs_dim = env.observation_space(env_params).shape[0]

		# --- Initialization ---
		if train_state is None:
			rng, _rng = jax.random.split(rng)

			# Transformer Init Shapes
			init_obs = jnp.zeros((2, obs_dim))
			init_memory = jnp.zeros((2, config.window_mem, config.num_layers, config.embed_size))
			init_mask = jnp.zeros((2, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_)

			network_params = network.init(_rng, init_memory, init_obs, init_mask)

			train_state = TrainState.create(
				apply_fn=network.apply,
				params=network_params,
				tx=tx,
			)

		rng, _rng = jax.random.split(rng)
		obsv, env_state = env.reset(_rng, env_params)
		env_state = env_state.replace(
			running_original_return=jnp.full(
				(config.num_envs,), current_original_return, dtype=jnp.float32
			)
		)

		# --- Initialize Transformer Memory State ---
		memories = jnp.zeros(
			(config.num_envs, config.window_mem, config.num_layers, config.embed_size)
		)
		memories_mask = jnp.zeros(
			(config.num_envs, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_
		)
		memories_mask_idx = jnp.zeros((config.num_envs,), dtype=jnp.int32) + (config.window_mem + 1)
		done = jnp.zeros((config.num_envs,), dtype=jnp.bool_)

		rng, _rng = jax.random.split(rng)

		# Current loop counter needed for memory index calculation
		init_step_env_currentloop = 0

		# Current update step counter
		init_update_step = 0

		initial_runner_state = (
			train_state,
			env_state,
			memories,
			memories_mask,
			memories_mask_idx,
			obsv,
			done,
			init_step_env_currentloop,
			init_update_step,
			_rng,
		)

		def _log_callback(metrics, step):
			# Unpack the tuple (now includes max)
			t_loss, v_loss, a_loss, ent, g_norm_mean, g_norm_max = metrics

			wandb.log(
				{
					"train/total_loss": t_loss,
					"train/value_loss": v_loss,
					"train/actor_loss": a_loss,
					"train/entropy": ent,
					"train/grad_norm_mean": g_norm_mean,
					"train/grad_norm_max": g_norm_max,  # <--- New
					"global_step": step,
				}
			)

		# --------------------------
		# The Transformer PPO update step
		# --------------------------
		def _update_step(runner_state, unused_scan_input):
			# === DEBUG: PRINT LEARNING RATE ===
			# # 1. Unpack just enough to get the optimizer state
			# _train_state = runner_state[0]

			# # 2. Extract the step count from Optax state.
			# # Optax states are trees; for Adam/Clip chains, the count is usually the first leaf.
			# _step_count = jax.tree_util.tree_leaves(_train_state.opt_state)[0]

			# # 3. Calculate current LR using the schedule function defined in the outer scope
			# _current_lr = linear_schedule(_step_count)

			# # 4. Print using JAX debug (Safe inside JIT/Scan)
			# # Use formatting to make it readable.
			# # We condition it to print only periodically if you want,
			# # but printing every update is usually fine.
			# jax.debug.print("Step: {x} | LR: {y:.8f}", x=_step_count, y=_current_lr)
			# ==================================
			# === A. COLLECT TRAJECTORIES ===
			def _env_step(runner_state, _):
				(
					train_state,
					env_state,
					memories,
					memories_mask,
					memories_mask_idx,
					last_obs,
					done,
					step_env_currentloop,
					update_step,
					rng,
				) = runner_state

				# 1. Reset memories mask if done
				memories_mask_idx = jnp.where(
					done, config.window_mem, jnp.clip(memories_mask_idx - 1, 0, config.window_mem)
				)
				memories_mask = jnp.where(
					done[:, None, None, None],
					jnp.zeros(
						(config.num_envs, config.num_heads, 1, config.window_mem + 1),
						dtype=jnp.bool_,
					),
					memories_mask,
				)

				# 2. Update memories mask with the potential additional step
				memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.window_mem + 1)
				memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
					config.num_heads, 1
				)
				memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

				# 3. Select Action
				rng, _rng = jax.random.split(rng)
				pi, value, memories_out = network.apply(
					train_state.params,
					memories,
					last_obs,
					memories_mask,
					method=network.model_forward_eval,
				)
				action = pi.sample(seed=_rng)
				log_prob = pi.log_prob(action)

				# 4. Update Cache: Roll memory and add new output
				memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

				# 5. Step Env
				rng, _rng = jax.random.split(rng)
				obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

				env_state = env_state.replace(
					running_original_return=jnp.full(
						(config.num_envs,), current_original_return, dtype=jnp.float32
					)
				)

				# 6. Compute memory indices for training
				memory_indices = jnp.arange(0, config.window_mem)[
					None, :
				] + step_env_currentloop * jnp.ones((config.num_envs, 1), dtype=jnp.int32)

				transition = Transition(
					done,
					action,
					value,
					reward,
					log_prob,
					memories_mask.squeeze(),
					memory_indices,
					last_obs,
					info,
				)

				carry = (
					train_state,
					env_state,
					memories,
					memories_mask,
					memories_mask_idx,
					obsv,
					done,
					step_env_currentloop + 1,
					update_step,
					rng,
				)
				return carry, (transition, memories_out)

			# Save previous memories to concatenate later (so first step of batch has context)
			memories_previous = runner_state[2]

			(final_state_carry), (traj_batch, memories_batch) = jax.lax.scan(
				_env_step, runner_state, None, config.num_steps
			)

			(
				train_state,
				final_env_state,
				final_memories,
				final_mask,
				final_mask_idx,
				final_obs,
				done,
				final_step_loop,
				update_step,
				rng,
			) = final_state_carry

			# === B. CALCULATE ADVANTAGES (GAE) ===
			# For GAE we need the value of the *next* state (final_obs)
			_, last_val, _ = network.apply(
				train_state.params,
				final_memories,
				final_obs,
				final_mask,
				method=network.model_forward_eval,
			)

			def _calculate_gae(traj_batch, last_val):
				def _get_advantages(carry, transition):
					gae, next_value = carry
					done, value, reward = transition.done, transition.value, transition.reward
					delta = reward + config.gamma * next_value * (1 - done) - value
					gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
					return (gae, value), gae

				_, advantages = jax.lax.scan(
					_get_advantages,
					(jnp.zeros_like(last_val), last_val),
					traj_batch,
					reverse=True,
					unroll=16,
				)
				return advantages, advantages + traj_batch.value

			advantages, targets = _calculate_gae(traj_batch, last_val)

			# Prepare scoring data (standard PPO interface for your calculator)
			# We strip out transformer specifics here because scoring doesn't need them
			scoring_traj = traj_batch.replace(
				obs=None, action=None, log_prob=None, memories_mask=None, memories_indices=None
			)
			# Explicitly set fields to None if the NamedTuple structure allows, or just reconstruct
			# Actually, Transition has new fields. The calculator expects `info` etc.
			# It should be fine as long as `scoring.py` accesses attributes by name.

			scoring_data = {"traj_batch": scoring_traj, "advantages": advantages}

			# NEW: Metric Logging for Original Task (Last Task ID)
			# The original task is always appended to the end, so its ID is num_tasks - 1
			original_task_idx = num_tasks - 1
			task_mask = traj_batch.info["task_id"] == original_task_idx
			# We only care about episodes that actually returned (finished)
			valid_mask = traj_batch.info["returned_episode"] * task_mask

			# # Calculate metrics only for the valid episodes of the original task
			# # We add epsilon to valid_mask.sum() to avoid division by zero if no original task episodes finished
			# metric = jax.tree.map(
			# 	lambda x: (x * valid_mask).sum() / (valid_mask.sum() + 1e-8),
			# 	traj_batch.info,
			# )

			# if config.debug and config.use_wandb:
			#     # Calculate cumulative steps based on global update count to ensure continuity
			# 	current_global_update = initial_global_update_step + update_step
			# 	current_env_steps = current_global_update * config.num_steps * config.num_envs

			# 	def callback(metric, global_update, env_steps):
			# 		to_log = create_log_dict(metric, config)
			# 		batch_log(global_update, to_log, config, env_steps)

			# 	jax.debug.callback(callback, metric, current_global_update, current_env_steps)

			# === C. UPDATE NETWORK (TRANSFORMER LOSS) ===

			# Concatenate previous memories so the first steps of the batch have context
			memories_batch = jnp.concatenate(
				[jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0
			)

			def _update_epoch(update_state, unused):
				def _update_minbatch(train_state, batch_info):
					traj_batch, memories_batch, advantages, targets = batch_info

					# advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

					def _loss_fn(params, traj_batch, memories_batch, gae, targets):
						# --- TRANSFORMER SPECIFIC: MEMORY BATCHING ---
						# Construct memory batch from indices
						memories_batch = batch_indices_select(
							memories_batch, traj_batch.memories_indices[:, :: config.window_grad]
						)
						memories_batch = batchify(memories_batch)

						# Create Mask for Window Grad
						memories_mask = traj_batch.memories_mask.reshape(
							(
								-1,
								config.window_grad,
							)
							+ traj_batch.memories_mask.shape[2:]
						)
						memories_mask = jnp.swapaxes(memories_mask, 1, 2)
						# Concatenate with 0s
						memories_mask = jnp.concatenate(
							(
								memories_mask,
								jnp.zeros(
									memories_mask.shape[:-1] + (config.window_grad - 1,),
									dtype=jnp.bool_,
								),
							),
							axis=-1,
						)
						# Roll
						memories_mask = roll_vmap(
							memories_mask, jnp.arange(0, config.window_grad), -1
						)

						# Reshape Obs and Batch
						obs = traj_batch.obs.reshape(
							(
								-1,
								config.window_grad,
							)
							+ traj_batch.obs.shape[2:]
						)
						traj_batch_r, targets_r, gae_r = jax.tree_util.tree_map(
							lambda x: jnp.reshape(x, (-1, config.window_grad) + x.shape[2:]),
							(traj_batch, targets, gae),
						)

						# Network Output (Train Mode)
						pi, value = network.apply(
							params,
							memories_batch,
							obs,
							memories_mask,
							method=network.model_forward_train,
						)
						log_prob = pi.log_prob(traj_batch_r.action)

						# Value Loss
						value_pred_clipped = traj_batch_r.value + (value - traj_batch_r.value).clip(
							-config.clip_eps, config.clip_eps
						)
						value_losses = jnp.square(value - targets_r)
						value_losses_clipped = jnp.square(value_pred_clipped - targets_r)
						value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

						# Actor Loss
						ratio = jnp.exp(log_prob - traj_batch_r.log_prob)
						gae_r = (gae_r - gae_r.mean()) / (gae_r.std() + 1e-8)
						loss_actor1 = ratio * gae_r
						loss_actor2 = (
							jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae_r
						)
						loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

						entropy = pi.entropy().mean()
						total_loss = (
							loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
						)
						return total_loss, (value_loss, loss_actor, entropy)

					grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
					(total_loss, (value_loss, loss_actor, entropy)), grads = grad_fn(
						train_state.params, traj_batch, memories_batch, advantages, targets
					)

					grad_norm = optax.global_norm(grads)
					train_state = train_state.apply_gradients(grads=grads)
					return train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm)

				(train_state, traj_batch, memories_batch, advantages, targets, update_step, rng) = (
					update_state
				)
				rng, _rng = jax.random.split(rng)

				# Batch Permutation
				permutation = jax.random.permutation(_rng, config.num_envs)
				batch = (traj_batch, memories_batch, advantages, targets)

				# Swap axes to (Envs, Steps, ...)
				batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
				# Shuffle envs
				shuffled_batch = jax.tree_util.tree_map(
					lambda x: jnp.take(x, permutation, axis=0), batch
				)

				# Create Minibatches
				minibatches = jax.tree_util.tree_map(
					lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
					shuffled_batch,
				)

				train_state, (total_loss, value_loss, loss_actor, entropy, grad_norm) = jax.lax.scan(_update_minbatch, train_state, minibatches)
				return (
					train_state,
					traj_batch,
					memories_batch,
					advantages,
					targets,
					update_step,
					rng,
				), (total_loss, value_loss, loss_actor, entropy, grad_norm)

			update_state = (
				train_state,
				traj_batch,
				memories_batch,
				advantages,
				targets,
				update_step,
				rng,
			)
			update_state, rl_info = jax.lax.scan(
				_update_epoch, update_state, None, config.update_epochs
			)

			losses_and_ent = rl_info[:4]
			grad_norms = rl_info[4]

			# 1. Calculate Means for losses (Standard)
			losses_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), losses_and_ent)

			# 2. Calculate Mean AND Max for grad_norm (Diagnostic)
			gn_mean = jnp.mean(grad_norms)
			gn_max = jnp.max(grad_norms)

			# 3. Pack it all up for the callback
			# Structure: (t_loss, v_loss, a_loss, ent, g_norm_mean, g_norm_max)
			metrics_to_log = (*losses_mean, gn_mean, gn_max)

			current_step = initial_global_update_step + update_step
			jax.debug.callback(_log_callback, metrics_to_log, current_step)
			

			# D. PREPARE FOR NEXT STEP
			train_state = update_state[0]
			rng = update_state[-1]
			# Reset loop counter to 0 for the next block of rollouts
			next_runner_state = (
				train_state,
				final_env_state,
				final_memories,
				final_mask,
				final_mask_idx,
				final_obs,
				done,
				0,
				update_step + 1,
				rng,
			)

			return next_runner_state, scoring_data

		# --- Run fixed-iteration training loop ---
		(final_runner_state, scan_scoring_data) = jax.lax.scan(
			_update_step, initial_runner_state, None, length=NUM_UPDATES
		)

		final_train_state = final_runner_state[0]

		# --- Process outputs for the "Smart Calculator" (IDENTICAL TO OLD CODE) ---
		k = config.scoring_window_updates
		scoring_window_data = jax.tree.map(lambda x: x[-k:], scan_scoring_data)

		# Flatten k and num_steps
		flat_traj = jax.tree.map(
			lambda x: x.reshape(-1, *x.shape[2:]), scoring_window_data["traj_batch"]
		)
		flat_advantages = scoring_window_data["advantages"].reshape(
			-1, *scoring_window_data["advantages"].shape[2:]
		)

		final_scoring_window_data = {"traj_batch": flat_traj, "advantages": flat_advantages}

		num_env_steps_done = NUM_UPDATES * config.num_envs * config.num_steps

		return {
			"train_state": final_train_state,
			"metrics": {
				"scoring_window_data": final_scoring_window_data,
				"num_updates_done": NUM_UPDATES,
				"num_env_steps_done": num_env_steps_done,
			},
		}

	return train


def make_eval(config, task_classes, num_training_updates, task_embeddings=None):
	"""Identical to make_train, but deletes the Policy/Value update step.
	It runs rollouts and GAE (for scoring) but returns the train_state UNCHANGED.
	"""
	# --- 1. Environment Setup (Same as Train) ---
	NUM_UPDATES = num_training_updates
	num_tasks = len(task_classes)
	static_env_params = StaticEnvParams()
	env_params = EnvParams(max_timesteps=4096)
	# env_params = EnvParams()

	# ... (Copy the Env setup logic from make_train exactly) ...
	if config.dicode_manager.mode != "reward":
		if task_embeddings is not None:
			embedding_size = task_embeddings.shape[1]
			base_env = MultiTaskMiniCraftaxEnv(
				task_classes,
				static_env_params,
				env_params,
				config.training.condition_on_task,
				conditioning_type="embedding",
				embedding_size=embedding_size,
				completion_bonus_scale=config.dicode_manager.completion_bonus_scale,
				completion_bonus_min=config.dicode_manager.completion_bonus_min,
			)
		else:
			base_env = MultiTaskMiniCraftaxEnv(
				task_classes,
				static_env_params,
				env_params,
				config.training.condition_on_task,
				completion_bonus_scale=config.dicode_manager.completion_bonus_scale,
				completion_bonus_min=config.dicode_manager.completion_bonus_min,
			)
	else:
		if task_embeddings is not None:
			embedding_size = task_embeddings.shape[1]
			base_env = MultiTaskMiniCraftaxEnvR(
				task_classes,
				static_env_params,
				env_params,
				config.training.condition_on_task,
				conditioning_type="embedding",
				embedding_size=embedding_size,
			)
		else:
			base_env = MultiTaskMiniCraftaxEnvR(
				task_classes, static_env_params, env_params, config.training.condition_on_task
			)

	# base_env = LogWrapper(base_env)

	env = MultiTaskOptimisticLogWrapperAllTasks(
		base_env,
		config.validation.num_envs,
		num_tasks,
		config.validation.optimistic_reset_ratio,
		task_embeddings,
	)
	env_params = env.default_params

	# env = CraftaxAugObsTrain()
	# env_params = env.default_params

	# env = MultiTaskOptimisticLogWrapperAllTasks(
	# 	env,
	# 	config.validation.num_envs,
	# 	1,
	# 	config.validation.optimistic_reset_ratio,
	# 	None,
	# )
	# env = LogWrapper(env)
	# env = OptimisticResetVecEnvWrapper(env, num_envs=config.validation.num_envs, reset_ratio=config.validation.optimistic_reset_ratio)

	# --- 2. Network Setup (Same as Train) ---
	network = ActorCriticTransformer(
		action_dim=env.action_space(env_params).n,
		activation=config.training.activation,
		hidden_layers=config.training.hidden_layers,
		encoder_size=config.training.embed_size,
		num_heads=config.training.num_heads,
		qkv_features=config.training.qkv_features,
		num_layers=config.training.num_layers,
		gating=config.training.gating,
		gating_bias=config.training.gating_bias,
	)

	# (No Optimizer needed for Eval, but we define 'tx' to satisfy TrainState creation if needed)
	tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))

	def eval_loop(rng, train_state=None):
		"""The JIT-compiled evaluation loop."""
		obs_dim = env.observation_space(env_params).shape[0]

		# --- Init (Same as Train) ---
		if train_state is None:
			# Should usually be passed in, but handle None case just in case
			rng, _rng = jax.random.split(rng)
			init_obs = jnp.zeros((2, obs_dim))
			init_memory = jnp.zeros(
				(
					2,
					config.training.window_mem,
					config.training.num_layers,
					config.training.embed_size,
				)
			)
			init_mask = jnp.zeros(
				(2, config.training.num_heads, 1, config.training.window_mem + 1), dtype=jnp.bool_
			)
			network_params = network.init(_rng, init_memory, init_obs, init_mask)
			train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

		rng, _rng = jax.random.split(rng)
		obsv, env_state = env.reset(_rng, env_params)

		# Transformer State Init
		memories = jnp.zeros(
			(
				config.validation.num_envs,
				config.training.window_mem,
				config.training.num_layers,
				config.training.embed_size,
			)
		)
		memories_mask = jnp.zeros(
			(
				config.validation.num_envs,
				config.training.num_heads,
				1,
				config.training.window_mem + 1,
			),
			dtype=jnp.bool_,
		)
		memories_mask_idx = jnp.zeros((config.validation.num_envs,), dtype=jnp.int32) + (
			config.training.window_mem + 1
		)
		done = jnp.zeros((config.validation.num_envs,), dtype=jnp.bool_)

		rng, _rng = jax.random.split(rng)
		init_runner_state = (
			train_state,
			env_state,
			memories,
			memories_mask,
			memories_mask_idx,
			obsv,
			done,
			0,
			_rng,
		)

		# --------------------------
		# The Evaluation Step (NO UPDATE)
		# --------------------------
		def _eval_update_step(runner_state, unused):
			# 1. Run the Environment Rollout (Same as Train)
			#    We use the exact same _env_step logic to ensure Transformer memory
			#    is handled correctly during rollout.

			def _env_step(carry, _):
				(
					train_state,
					env_state,
					memories,
					memories_mask,
					memories_mask_idx,
					last_obs,
					done,
					step_env_currentloop,
					rng,
				) = carry

				# ... (Copy logic from make_train's _env_step) ...
				memories_mask_idx = jnp.where(
					done,
					config.training.window_mem,
					jnp.clip(memories_mask_idx - 1, 0, config.training.window_mem),
				)
				memories_mask = jnp.where(
					done[:, None, None, None],
					jnp.zeros(
						(
							config.validation.num_envs,
							config.training.num_heads,
							1,
							config.training.window_mem + 1,
						),
						dtype=jnp.bool_,
					),
					memories_mask,
				)
				memories_mask_idx_ohot = jax.nn.one_hot(
					memories_mask_idx, config.training.window_mem + 1
				)
				memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
					config.training.num_heads, 1
				)
				memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

				rng, _rng = jax.random.split(rng)
				# Note: model_forward_eval
				pi, value, memories_out = network.apply(
					train_state.params,
					memories,
					last_obs,
					memories_mask,
					method=network.model_forward_eval,
				)
				action = pi.sample(seed=_rng)
				log_prob = pi.log_prob(action)

				memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

				rng, _rng = jax.random.split(rng)
				obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

				memory_indices = jnp.arange(0, config.training.window_mem)[
					None, :
				] + step_env_currentloop * jnp.ones(
					(config.validation.num_envs, 1), dtype=jnp.int32
				)

				transition = Transition(
					done,
					action,
					value,
					reward,
					log_prob,
					memories_mask.squeeze(),
					memory_indices,
					last_obs,
					info,
				)
				return (
					train_state,
					env_state,
					memories,
					memories_mask,
					memories_mask_idx,
					obsv,
					done,
					step_env_currentloop + 1,
					rng,
				), transition

			(final_state_carry), traj_batch = jax.lax.scan(
				_env_step, runner_state, None, config.validation.num_steps
			)

			(
				train_state,
				final_env_state,
				final_memories,
				final_mask,
				final_mask_idx,
				final_obs,
				done,
				final_step_loop,
				rng,
			) = final_state_carry

			# 2. Calculate Advantages (GAE)
			#    We STILL need this because your 'Smart Calculator' uses advantages
			#    to compute PVL (Positive Value Loss) scores.
			_, last_val, _ = network.apply(
				train_state.params,
				final_memories,
				final_obs,
				final_mask,
				method=network.model_forward_eval,
			)
			# last_val = last_val.squeeze(0)

			def _calculate_gae(traj_batch, last_val):
				def _get_advantages(carry, transition):
					gae, next_value = carry
					done, value, reward = transition.done, transition.value, transition.reward
					delta = reward + config.training.gamma * next_value * (1 - done) - value
					gae = (
						delta
						+ config.training.gamma * config.training.gae_lambda * (1 - done) * gae
					)
					return (gae, value), gae

				_, advantages = jax.lax.scan(
					_get_advantages,
					(jnp.zeros_like(last_val), last_val),
					traj_batch,
					reverse=True,
					unroll=16,
				)
				return advantages

			advantages = _calculate_gae(traj_batch, last_val)

			# 3. Prepare Data for Output
			scoring_traj = traj_batch.replace(
				obs=None, action=None, log_prob=None, memories_mask=None, memories_indices=None
			)

			scoring_data = {"traj_batch": scoring_traj, "advantages": advantages}

			# 4. CRITICAL DIFFERENCE: NO UPDATE STEP
			# We do not run _update_epoch. We do not calculate gradients.
			# We simply return the state (ready for next rollout) and the data.

			# Reset loop counter to 0
			next_runner_state = (
				train_state,
				final_env_state,
				final_memories,
				final_mask,
				final_mask_idx,
				final_obs,
				done,
				0,
				rng,
			)

			return next_runner_state, scoring_data

		# Run the Scan
		(final_runner_state, scan_scoring_data) = jax.lax.scan(
			_eval_update_step, init_runner_state, None, length=NUM_UPDATES
		)

		final_train_state = final_runner_state[0]

		# Process for Calculator
		# k = config.scoring_window_updates
		# scoring_window_data = jax.tree.map(lambda x: x[-k:], scan_scoring_data)
		scoring_window_data = scan_scoring_data

		flat_traj = jax.tree.map(
			lambda x: x.reshape(-1, *x.shape[2:]), scoring_window_data["traj_batch"]
		)
		flat_advantages = scoring_window_data["advantages"].reshape(
			-1, *scoring_window_data["advantages"].shape[2:]
		)

		final_scoring_window_data = {"traj_batch": flat_traj, "advantages": flat_advantages}

		num_env_steps_done = NUM_UPDATES * config.validation.num_envs * config.validation.num_steps

		return {
			"train_state": final_train_state,  # Unchanged!
			"metrics": {
				"scoring_window_data": final_scoring_window_data,
				"num_updates_done": NUM_UPDATES,
				"num_env_steps_done": num_env_steps_done,
			},
		}

	return eval_loop


# =================================================================
# === TOP-LEVEL API (UNCHANGED) ===================================
# =================================================================


def run_training_session(
	config,
	rng,
	task_classes,
	num_training_updates,
	task_embeddings=None,
	train_state=None,
	task_distribution_proportions=None,
	global_update_step=0,
	current_original_return=0.0,
):
	config_t = config.training
	train_fn = make_train(
		config_t,
		task_classes,
		num_training_updates,
		task_embeddings,
		task_distribution_proportions,
		global_update_step,
	)
	train_jit = jax.jit(train_fn)
	print("JIT compiling and running training session (Transformer)...")
	start_time = time.time()
	results = train_jit(rng, train_state, current_original_return)
	print(f"Session finished in {time.time() - start_time:.2f} seconds.")
	return results


def run_evaluation_rollouts(
	config, rng, task_classes, num_training_updates, task_embeddings=None, train_state=None
):
	if train_state is None:
		raise ValueError("run_evaluation_rollouts requires a valid train_state.")
	eval_fn = make_eval(config, task_classes, num_training_updates, task_embeddings)
	eval_jit = jax.jit(eval_fn)
	print("JIT compiling and running evaluation rollouts (Transformer)...")
	start_time = time.time()
	results = eval_jit(rng, train_state)
	print(f"Session finished in {time.time() - start_time:.2f} seconds.")
	return results
