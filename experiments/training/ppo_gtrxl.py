"""Standalone PPO-GTrXL trainer on Craftax — the non-curriculum reference baseline.

This is the script behind the "PPO-GTrXL" rows in the paper. It trains a Gated Transformer-XL
policy on the default Craftax distribution, sampling a fresh seed each episode without reuse.

Not to be confused with `src/dicode/ppo_tr.py`, which is DiCode's *internal* multitask trainer
over MiniCraftax tasks and is not runnable as a standalone baseline.

Config comes from `conf/config.yaml` (hydra), whose `training` group defaults to
`conf/training/default.yaml` — the paper's hyperparameters (Table 6). The learning rate anneals
linearly from `lr` to 0; the `min_lr` floor used by DiCode is deliberately not applied here.

    python experiments/training/ppo_gtrxl.py seed=<SEED>
"""

import csv
import os
import time
from collections.abc import Sequence
from typing import NamedTuple

import distrax
import flax.linen as nn
import hydra
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from dicode.utils.logz.batch_logging import batch_log, create_log_dict
from dicode.transformer.transformerXL import Transformer
from dicode.wrappers import (
	LogWrapper,
	OptimisticResetVecEnvWrapper,
)


class ActorCriticTransformer(nn.Module):
	action_dim: Sequence[int]
	activation: str
	hidden_layers: int
	encoder_size: int
	num_heads: int
	qkv_features: int
	num_layers: int
	gating: bool = False
	gating_bias: float = 0.0

	def setup(self):
		# USE SETUP AND DIFFERENT FUNCTIONS BECAUSE THE TRAIN IS DIFFERENT FROM EVAL ( as we query just one step in train and don't cache memory in eval)

		if self.activation == "relu":
			self.activation_fn = nn.relu
		else:
			self.activation_fn = nn.tanh

		self.transformer = Transformer(
			encoder_size=self.encoder_size,
			num_heads=self.num_heads,
			qkv_features=self.qkv_features,
			num_layers=self.num_layers,
			gating=self.gating,
			gating_bias=self.gating_bias,
		)

		self.actor_ln1 = nn.Dense(
			self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
		)
		self.actor_ln2 = nn.Dense(
			self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
		)
		self.actor_out = nn.Dense(
			self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
		)

		self.critic_ln1 = nn.Dense(
			self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
		)
		self.critic_ln2 = nn.Dense(
			self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
		)
		self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

	def __call__(self, memories, obs, mask):
		x, memory_out = self.transformer(memories, obs, mask)

		actor_mean = self.actor_ln1(x)
		actor_mean = self.activation_fn(actor_mean)
		actor_mean = self.actor_ln2(actor_mean)
		actor_mean = self.activation_fn(actor_mean)
		actor_mean = self.actor_out(actor_mean)
		pi = distrax.Categorical(logits=actor_mean)

		critic = self.critic_ln1(x)
		critic = self.activation_fn(critic)
		critic = self.critic_ln2(critic)
		critic = self.activation_fn(critic)
		critic = self.critic_out(critic)

		return pi, jnp.squeeze(critic, axis=-1), memory_out

	def model_forward_eval(self, memories, obs, mask):
		"""Used during environment rollout (single timestep of obs). And return the memory"""
		x, memory_out = self.transformer.forward_eval(memories, obs, mask)

		actor_mean = self.actor_ln1(x)
		actor_mean = self.activation_fn(actor_mean)
		actor_mean = self.actor_ln2(actor_mean)
		actor_mean = self.activation_fn(actor_mean)
		actor_mean = self.actor_out(actor_mean)
		pi = distrax.Categorical(logits=actor_mean)

		critic = self.critic_ln1(x)
		critic = self.activation_fn(critic)
		critic = self.critic_ln2(critic)
		critic = self.activation_fn(critic)
		critic = self.critic_out(critic)

		return pi, jnp.squeeze(critic, axis=-1), memory_out

	def model_forward_train(self, memories, obs, mask):
		"""Used during training: a window of observation is sent. And don't return the memory"""
		x = self.transformer.forward_train(memories, obs, mask)

		actor_mean = self.actor_ln1(x)
		actor_mean = self.activation_fn(actor_mean)
		actor_mean = self.actor_ln2(actor_mean)
		actor_mean = self.activation_fn(actor_mean)
		actor_mean = self.actor_out(actor_mean)
		pi = distrax.Categorical(logits=actor_mean)

		critic = self.critic_ln1(x)
		critic = self.activation_fn(critic)
		critic = self.critic_ln2(critic)
		critic = self.activation_fn(critic)
		critic = self.critic_out(critic)
		return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
	done: jnp.ndarray
	action: jnp.ndarray
	value: jnp.ndarray
	reward: jnp.ndarray
	log_prob: jnp.ndarray
	memories_mask: jnp.ndarray
	memories_indices: jnp.ndarray
	obs: jnp.ndarray
	info: jnp.ndarray


indices_select = lambda x, y: x[y]
batch_indices_select = jax.vmap(indices_select)
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


def save_runtime_measurements(output_dir, main_jax_compile_seconds, total_run_seconds):
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, "ppo_tr_timings.csv")
	rows = [
		{"metric": "main_jax_compilation_seconds", "seconds": main_jax_compile_seconds},
		{"metric": "total_run_seconds", "seconds": total_run_seconds},
	]
	with open(output_path, "w", newline="") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=["metric", "seconds"])
		writer.writeheader()
		writer.writerows(rows)
	return output_path


def make_train(config):
	env = CraftaxSymbolicEnvNoAutoReset()
	env_params = env.default_params

	env = LogWrapper(env)
	env = OptimisticResetVecEnvWrapper(
		env,
		config.num_envs,
		config.optimistic_reset_ratio,
	)

	# INIT NETWORK
	network = ActorCriticTransformer(
		action_dim=env.action_space(env_params).n,
		activation=config.activation,
		encoder_size=config.embed_size,
		hidden_layers=config.hidden_layers,
		num_heads=config.num_heads,
		qkv_features=config.qkv_features,
		num_layers=config.num_layers,
		gating=config.gating,
		gating_bias=config.gating_bias,
	)

	def train(start_runner_state, num_updates_to_run):
		# TRAIN LOOP
		def _update_step(runner_state, unused):
			# COLLECT TRAJECTORIES
			def _env_step(runner_state, unused):
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

				# reset memories mask and mask idx in cask of done otherwise mask will consider one more stepif not filled (if filled=
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

				# Update memories mask with the potential additional step taken into account at this step
				memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config.window_mem + 1)
				memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
					config.num_heads, 1
				)
				memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

				# SELECT ACTION
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

				# ADD THE CACHED ACTIVATIONS IN MEMORIES FOR NEXT STEP
				memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

				# STEP ENV
				rng, _rng = jax.random.split(rng)
				# rng_step = jax.random.split(_rng, config["NUM_ENVS"])
				# obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
				#    rng_step, env_state, action, env_params
				# )
				obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

				# COMPUTE THE INDICES OF THE FINAL MEMORIES THAT ARE TAKEN INTO ACCOUNT IN THIS STEP
				# not forgeeting that we will concatenate the previous WINDOW_MEM to the NUM_STEPS so that even the first step will use some cached memory.
				# previous without this is attend to 0 which are masked but with reset happening if we start the num_steps loop during good to keep memory from previous
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
				runner_state = (
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
				return runner_state, (transition, memories_out)

			# also copy the first memories in memories_previous before the new rollout to concatenate previous memories with new steps so that first steps of new have memories
			memories_previous = runner_state[2]

			# SCAN THE STEP TO GET THE TRANSITIONS AND CACHED MEMORIES
			runner_state, (traj_batch, memories_batch) = jax.lax.scan(
				_env_step, runner_state, None, config.num_steps
			)

			# CALCULATE ADVANTAGE
			(
				train_state,
				env_state,
				memories,
				memories_mask,
				memories_mask_idx,
				last_obs,
				done,
				_,
				update_step,
				rng,
			) = runner_state
			_, last_val, _ = network.apply(
				train_state.params,
				memories,
				last_obs,
				memories_mask,
				method=network.model_forward_eval,
			)

			def _calculate_gae(traj_batch, last_val):
				def _get_advantages(gae_and_next_value, transition):
					gae, next_value = gae_and_next_value
					done, value, reward = (
						transition.done,
						transition.value,
						transition.reward,
					)
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

			# UPDATE NETWORK
			def _update_epoch(update_state, unused):
				def _update_minbatch(train_state, batch_info):
					traj_batch, memories_batch, advantages, targets = batch_info

					def _loss_fn(params, traj_batch, memories_batch, gae, targets):
						# USE THE CACHED MEMORIES ONLY FROM THE FIRST STEP OF A WINDOW GRAD Because all other will be computed again here.
						# construct the memory batch from memory indices
						memories_batch = batch_indices_select(
							memories_batch, traj_batch.memories_indices[:, :: config.window_grad]
						)
						memories_batch = batchify(memories_batch)

						# CREATE THE MASK FOR WINDOW GRAD (have to take the one from the batch and roll them to match the steps it attends
						memories_mask = traj_batch.memories_mask.reshape(
							(
								-1,
								config.window_grad,
							)
							+ traj_batch.memories_mask.shape[2:]
						)
						memories_mask = jnp.swapaxes(memories_mask, 1, 2)
						# concatenate with 0s to fill before the roll
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
						# roll of different value for each step to match the right
						memories_mask = roll_vmap(
							memories_mask, jnp.arange(0, config.window_grad), -1
						)

						# RESHAPE
						obs = traj_batch.obs
						obs = obs.reshape(
							(
								-1,
								config.window_grad,
							)
							+ obs.shape[2:]
						)

						traj_batch, targets, gae = jax.tree_util.tree_map(
							lambda x: jnp.reshape(x, (-1, config.window_grad) + x.shape[2:]),
							(traj_batch, targets, gae),
						)

						# NETWORK OUTPUT
						pi, value = network.apply(
							params,
							memories_batch,
							obs,
							memories_mask,
							method=network.model_forward_train,
						)

						log_prob = pi.log_prob(traj_batch.action)

						# CALCULATE VALUE LOSS
						value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
							-config.clip_eps, config.clip_eps
						)
						value_losses = jnp.square(value - targets)
						value_losses_clipped = jnp.square(value_pred_clipped - targets)
						value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

						# CALCULATE ACTOR LOSS
						ratio = jnp.exp(log_prob - traj_batch.log_prob)
						gae = (gae - gae.mean()) / (gae.std() + 1e-8)
						loss_actor1 = ratio * gae
						loss_actor2 = (
							jnp.clip(
								ratio,
								1.0 - config.clip_eps,
								1.0 + config.clip_eps,
							)
							* gae
						)
						loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
						loss_actor = loss_actor.mean()
						entropy = pi.entropy().mean()

						total_loss = (
							loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
						)
						return total_loss, (value_loss, loss_actor, entropy)

					grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
					total_loss, grads = grad_fn(
						train_state.params, traj_batch, memories_batch, advantages, targets
					)
					train_state = train_state.apply_gradients(grads=grads)
					return train_state, total_loss

				train_state, traj_batch, memories_batch, advantages, targets, rng = update_state
				rng, _rng = jax.random.split(rng)
				# batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
				assert config.num_steps % config.window_grad == 0, (
					"NUM_STEPS should be divi by WINDOW_GRAD to properly batch the window_grad"
				)

				# PERMUTE ALONG THE NUM_ENVS ONLY NOT TO LOOSE TRACK FROM TEMPORAL
				permutation = jax.random.permutation(_rng, config.num_envs)
				batch = (traj_batch, memories_batch, advantages, targets)
				batch = jax.tree_util.tree_map(
					lambda x: jnp.swapaxes(x, 0, 1),
					batch,
				)
				shuffled_batch = jax.tree_util.tree_map(
					lambda x: jnp.take(x, permutation, axis=0), batch
				)

				# either create memory batch here but might be big  or send all the memeory to loss and do the things with the index in the loss
				minibatches = jax.tree_util.tree_map(
					lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
					shuffled_batch,
				)

				train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)

				update_state = (train_state, traj_batch, memories_batch, advantages, targets, rng)
				return update_state, total_loss

			# ADD PREVIOUS WINDOW_MEM To the current NUM_STEPS SO THAT FIRST STEPS USE MEMORIES FROM PREVIOUS
			# might be a better place to add the previous memory to the traj batch to make it faster ???
			# or another solution is to not add it but in training means that the first element might not look at info
			memories_batch = jnp.concatenate(
				[jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0
			)

			# CRAFTAX ONLY
			metric = jax.tree.map(
				lambda x: (x * traj_batch.info["returned_episode"]).sum()
				/ traj_batch.info["returned_episode"].sum(),
				traj_batch.info,
			)
			# metric=jax.tree_map(lambda x: x.mean(),metric)

			if config.debug and config.use_wandb:
				global_env_steps = update_step * config.num_steps * config.num_envs

				def callback(metric, update_step, env_steps):
					to_log = create_log_dict(metric, config)
					batch_log(update_step, to_log, config, env_steps)

				jax.debug.callback(callback, metric, update_step, global_env_steps)

			update_state = (train_state, traj_batch, memories_batch, advantages, targets, rng)

			# TRAIN LOOP
			update_state, loss_info = jax.lax.scan(
				_update_epoch, update_state, None, config.update_epochs
			)
			train_state = update_state[0]
			rng = update_state[-1]
			runner_state = (
				train_state,
				env_state,
				memories,
				memories_mask,
				memories_mask_idx,
				last_obs,
				done,
				0,
				update_step + 1,
				rng,
			)
			return runner_state, metric

		# The scan now starts from the passed-in state
		# and runs for `num_updates_to_run` steps.
		runner_state, metric = jax.lax.scan(
			_update_step, start_runner_state, None, num_updates_to_run
		)
		# Return the final state and metrics directly
		return runner_state, metric

	return train


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="config")
def main(config):
	tags= ["PPO-TR"]
	use_wandb = config.use_wandb
	if config.use_wandb:
		# Convert config to plain dictionary
		config_dict = {k: v for k, v in config.items() if not k.startswith("_")}
		wandb.init(
			project=config.wandb_project,
			entity=config.wandb_entity,
			tags=tags,
			config=config_dict,
			name=config.training.env_name + "-PPO_Tr-" + str(int(config.training.total_timesteps // 1e6)) + "M",
		)

	if config.use_wandb:
		# Define our custom x-axes
		wandb.define_metric("update_step")
		wandb.define_metric("global_env_steps")

		# Set default x-axis for all other metrics
		wandb.define_metric("*", step_metric="global_env_steps")

	rng = jax.random.PRNGKey(config.seed)

	# === STATE INITIALIZATION ===
	# We moved this logic from make_train/train to main

	# 1. INIT ENV (Needed for network init)
	env = CraftaxSymbolicEnvNoAutoReset()
	env_params = env.default_params
	# We'll wrap the env later, just need the base env for spaces

	config = OmegaConf.merge(config.training, {"use_wandb": use_wandb})

	# 2. INIT LR SCHEDULE (Needed for optimizer init)
	NUM_UPDATES = int(config.total_timesteps // config.num_steps // config.num_envs)

	def linear_schedule(count):
		frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / NUM_UPDATES
		return config.lr * frac

	# def linear_schedule(count):
	# 	frac = (
	# 		1.0 - (count // (config.num_minibatches * config.update_epochs)) / NUM_UPDATES
	# 	)
	# 	return config.min_lr + (config.lr - config.min_lr) * frac

	# 3. INIT NETWORK
	network = ActorCriticTransformer(
		action_dim=env.action_space(env_params).n,
		activation=config.activation,
		encoder_size=config.embed_size,
		hidden_layers=config.hidden_layers,
		num_heads=config.num_heads,
		qkv_features=config.qkv_features,
		num_layers=config.num_layers,
		gating=config.gating,
		gating_bias=config.gating_bias,
	)

	rng, _rng = jax.random.split(rng)
	init_obs = jnp.zeros((2, env.observation_space(env_params).shape[0]))
	init_memory = jnp.zeros(
		(
			2,
			config.window_mem,
			config.num_layers,
			config.embed_size,
		)
	)
	init_mask = jnp.zeros(
		(2, config.num_heads, 1, config.window_mem + 1),
		dtype=jnp.bool_,
	)
	network_params = network.init(_rng, init_memory, init_obs, init_mask)

	# 4. INIT OPTIMIZER & TRAIN STATE
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

	# This is our initial state
	train_state = TrainState.create(
		apply_fn=network.apply,
		params=network_params,
		tx=tx,
	)

	# === CHECKPOINT MANAGER SETUP ===

	ckpt_dir = os.path.join(os.getcwd(), config.checkpoint_dir)
	orbax_checkpointer = ocp.PyTreeCheckpointer()
	ckpt_options = ocp.CheckpointManagerOptions(
		max_to_keep=config.max_checkpoints_to_keep,
		keep_period=config.checkpoint_keep_period,
		create=True
	)
	ckpt_manager = ocp.CheckpointManager(ckpt_dir, orbax_checkpointer, options=ckpt_options)

	start_update = 0

	# === CHECKPOINT LOADING LOGIC ===

	latest_step = ckpt_manager.latest_step()
	if config.load_checkpoint and latest_step is not None:
		print(f"Loading checkpoint from step {latest_step}...")

		# Define the structure of what we expect to load
		abstract_save_data = {"train_state": train_state, "update_step": 0, "rng": rng}

		restored_data = ckpt_manager.restore(latest_step, args=ocp.RestoreArgs(abstract_save_data))

		train_state = restored_data["train_state"]
		start_update = restored_data["update_step"]
		rng = restored_data["rng"]  # Restore RNG for deterministic resume

		print(f"Successfully loaded. Resuming from update {start_update}.")
	else:
		print("Starting training from scratch.")

	# 1. INIT ENV & WRAPPERS
	# (We re-create the env here, as it's not part of the saved state)
	env = CraftaxSymbolicEnvNoAutoReset()
	env_params = env.default_params
	env = LogWrapper(env)
	env = OptimisticResetVecEnvWrapper(
		env,
		config.num_envs,
		config.optimistic_reset_ratio,
	)

	# 2. RESET ENV
	rng, _rng = jax.random.split(rng)
	obsv, env_state = env.reset(_rng, env_params)

	# 3. INIT MEMORIES & MASKS
	memories = jnp.zeros(
		(
			config.num_envs,
			config.window_mem,
			config.num_layers,
			config.embed_size,
		)
	)
	memories_mask = jnp.zeros(
		(config.num_envs, config.num_heads, 1, config.window_mem + 1),
		dtype=jnp.bool_,
	)
	# memories +1 bc will remove one
	memories_mask_idx = jnp.zeros((config.num_envs,), dtype=jnp.int32) + (config.window_mem + 1)
	done = jnp.zeros((config.num_envs,), dtype=jnp.bool_)
	rng, _rng = jax.random.split(rng)

	# 4. ASSEMBLE THE RUNNER_STATE TUPLE
	# This must match the order expected by _update_step
	current_runner_state = (
		train_state,
		env_state,
		memories,
		memories_mask,
		memories_mask_idx,
		obsv,
		done,
		0,  # step_env_currentloop
		start_update,  # update_step
		_rng,  # rng
	)

	# === CREATE JIT FUNCTION & OUTER LOOP ===

	# 1. Create the JIT-compiled training "chunk" function
	train_chunk_fn = make_train(config)
	train_chunk_jit = jax.jit(train_chunk_fn, static_argnums=1)

	# 2. Calculate update counts
	# (NUM_UPDATES was defined earlier, but we'll redefine it as num_updates
	#  just in case, or you can reuse the one from the LR schedule)
	num_updates = int(config.total_timesteps // config.num_steps // config.num_envs)
	checkpoint_interval = config.checkpoint_interval_updates

	print(f"Starting training... Total updates: {num_updates}. Resuming from {start_update}.")
	print(f"Checkpoints will be saved every {checkpoint_interval} updates.")

	total_run_start = time.perf_counter()
	total_run_seconds = 0.0
	main_jax_compile_time = 0.0
	main_train_chunk_compiled = None
	compile_start = None
	runtime_output_path = None
	main_updates_to_run = int(min(checkpoint_interval, num_updates - start_update))
	current_update = start_update
	run_completed = False

	try:
		if main_updates_to_run > 0:
			print(f"JIT compiling main training chunk for {main_updates_to_run} updates...")
			compile_start = time.perf_counter()
			main_train_chunk_compiled = train_chunk_jit.lower(
				current_runner_state, main_updates_to_run
			).compile()
			main_jax_compile_time = time.perf_counter() - compile_start
			compile_start = None
			print(f"Main JAX compilation time: {main_jax_compile_time:.2f} seconds")
		else:
			print("No training updates remaining. Skipping JAX compilation timing.")

		# 3. Run the outer Python loop
		while current_update < num_updates:
			# Determine how many updates to run in this chunk
			updates_to_run = int(min(checkpoint_interval, num_updates - current_update))
			if updates_to_run <= 0:
				break

			print(f"--- Running updates {current_update} to {current_update + updates_to_run} ---")

			# Run one JIT-compiled chunk of training
			# Reuse the explicitly compiled executable for the main chunk size.
			if updates_to_run == main_updates_to_run and main_train_chunk_compiled is not None:
				current_runner_state, metrics = main_train_chunk_compiled(current_runner_state)
			else:
				# Smaller trailing chunks may trigger an additional compile because
				# `updates_to_run` is static in the JIT signature.
				current_runner_state, metrics = train_chunk_jit(current_runner_state, updates_to_run)

			# Get the new update count from the returned state
			current_update = int(current_runner_state[8])  # (..., update_step, rng)

			# --- SAVING LOGIC ---
			print(f"Saving checkpoint at update step {current_update}...")

			# Extract the state components to save
			train_state_to_save = current_runner_state[0]  # train_state
			rng_to_save = current_runner_state[-1]  # rng

			save_data = {
				"train_state": train_state_to_save,
				"update_step": current_update,
				"rng": rng_to_save,
			}

			# Save the checkpoint
			ckpt_manager.save(current_update, items=save_data)

			# Wait for saving to finish
			ckpt_manager.wait_until_finished()
			print(f"Checkpoint {current_update} saved.")

			# Persist timings after each completed chunk so the latest elapsed
			# runtime survives later failures or interruptions.
			runtime_output_path = save_runtime_measurements(
				os.path.join(os.getcwd(), "runtime_analysis"),
				main_jax_compile_time,
				time.perf_counter() - total_run_start,
			)

			# You can log metrics to wandb here if you want
			# if config.use_wandb:
			#     wandb.log({"update": current_update, ...})

		run_completed = True
	finally:
		if compile_start is not None and main_jax_compile_time == 0.0:
			main_jax_compile_time = time.perf_counter() - compile_start
		total_run_seconds = time.perf_counter() - total_run_start

		try:
			runtime_output_path = save_runtime_measurements(
				os.path.join(os.getcwd(), "runtime_analysis"),
				main_jax_compile_time,
				total_run_seconds,
			)
			print(f"Saved runtime measurements to {runtime_output_path}")
		except Exception as timing_save_error:
			print(f"Warning: Failed to save runtime measurements. {timing_save_error}")

		if use_wandb and wandb.run is not None:
			try:
				wandb.summary["system/main_jax_compilation_seconds"] = main_jax_compile_time
				wandb.summary["system/total_run_seconds"] = total_run_seconds
			except Exception as wandb_error:
				print(f"Warning: Failed to update W&B timing summary. {wandb_error}")

	if run_completed:
		print("--- Training Complete ---")
		print(f"Main JAX compilation time: {main_jax_compile_time:.2f} seconds")
		print(f"Total time: {total_run_seconds:.2f} seconds")
	total_steps = (current_update - start_update) * config.num_steps * config.num_envs
	if total_run_seconds > 0:
		print(f"SPS: {total_steps / total_run_seconds:.2f}")

	# # Final save
	# if "save_data" in locals():  # Only save if we ran at least one loop
	# 	ckpt_manager.save(current_update, items=save_data, force=True)
	# 	ckpt_manager.wait_until_finished()
	# 	print(f"Final checkpoint {current_update} saved.")


if __name__ == "__main__":
	main()
