import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from dicode.network import ActorCriticTransformer, Transition
from dicode.wrappers import BatchEnvWrapper
from minicraftax.envs.craftax import CraftaxAugObsTrain
# --- 2. Transformer Network Class ---
# Imported from dicode.network


def make_evaluate(config, env, env_params):
	num_envs = config.evaluation.num_envs
	num_steps = config.evaluation.num_steps
	def evaluate(train_state, rng):
		network = ActorCriticTransformer(
			action_dim=env.action_space(env_params).n,
			activation=config.training.activation,
			hidden_layers=config.training.hidden_layers,  # Mapping layer_size to hidden_layers
			encoder_size=config.training.embed_size,  # Mapping embedding_size to encoder_size/embed_size
			num_heads=config.training.num_heads,
			qkv_features=config.training.qkv_features,
			num_layers=config.training.num_layers,
			gating=config.training.gating,
			gating_bias=config.training.gating_bias,
		)

		rng, reset_rng = jax.random.split(rng)
		obsv, env_state = env.reset(reset_rng, env_params)

		# --- 3. Determine Info Structure (Crucial Step) ---
		# We need to know what 'info' looks like to create a zero-filled accumulator.
		# We use eval_shape to do this without actually running the computation.
		def get_info_shape():
			dummy_action = jnp.zeros((num_envs,), dtype=jnp.int32)
			_, _, _, _, info = env.step(reset_rng, env_state, dummy_action, env_params)
			return info
		
		info_structure = jax.eval_shape(get_info_shape)

		# Initialize accumulated_stats with zeros matching info_structure
		accumulated_stats = jax.tree.map(
			lambda x: jnp.zeros((num_envs,) + x.shape[1:], dtype=jnp.float32), 
			info_structure
		)

		# Transformer State Init
		memories = jnp.zeros(
			(
				num_envs,
				config.training.window_mem,
				config.training.num_layers,
				config.training.embed_size,
			)
		)
		memories_mask = jnp.zeros(
			(
				num_envs,
				config.training.num_heads,
				1,
				config.training.window_mem + 1,
			),
			dtype=jnp.bool_,
		)
		memories_mask_idx = jnp.zeros((num_envs,), dtype=jnp.int32) + (
			config.training.window_mem + 1
		)

		finished_mask = jnp.zeros((num_envs,), dtype=jnp.bool_)
		accumulated_reward = jnp.zeros((num_envs,), dtype=jnp.float32)
		accumulated_length = jnp.zeros((num_envs,), dtype=jnp.float32)
		done_prev = jnp.zeros((num_envs,), dtype=jnp.bool_)

		init_runner_state = (
			train_state,
			env_state,
			memories,
			memories_mask,
			memories_mask_idx,
			obsv,
			done_prev,
			0,
			finished_mask,
			accumulated_reward,
			accumulated_length,
			accumulated_stats,
			rng,
		)

		# --------------------------
		# The Evaluation Step (NO UPDATE)
		# --------------------------

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
				finished_mask,
				acc_reward,
				acc_length,
				acc_stats,
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
						num_envs,
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
			pi, _, memories_out = network.apply(
				train_state.params,
				memories,
				last_obs,
				memories_mask,
				method=network.model_forward_eval,
			)
			action = pi.sample(seed=_rng)
			# log_prob = pi.log_prob(action)

			memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

			rng, step_rng = jax.random.split(rng)
			next_obsv, next_env_state, reward, next_done, info = env.step(step_rng, env_state, action, env_params)

			# memory_indices = jnp.arange(0, config.training.window_mem)[
			# 	None, :
			# ] + step_env_currentloop * jnp.ones((config.evaluation.num_envs, 1), dtype=jnp.int32)

			# transition = Transition(
			# 	done,
			# 	action,
			# 	value,
			# 	reward,
			# 	log_prob,
			# 	memories_mask.squeeze(),
			# 	memory_indices,
			# 	last_obs,
			# 	info,
			# )

			# --- METRIC ACCUMULATION LOGIC ---
			
			# 1. Calculate Mask: We only accept data if the env is NOT finished yet.
			# Convert bool mask to float (0.0 or 1.0) for multiplication
			active_mask = (1.0 - finished_mask.astype(jnp.float32))
			
			# 2. Accumulate Reward
			new_acc_reward = acc_reward + (reward * active_mask)
			new_acc_length = acc_length + (1 * active_mask).astype(jnp.int32)

			# 3. Accumulate Info/Achievements
			# We iterate over every key in 'info' (Achievements, discount, etc.)
			# If info contains {k: v}, we do: acc[k] += v * active_mask
			# Since 'v' is already (v * done) inside the env, this effectively captures
			# the value at the moment of termination and ignores all subsequent zeros.
			def accumulate_leaf(acc, new_val):
				# Ensure dimensions match for broadcasting if necessary
				# Usually info values are (num_envs,), so direct mul is fine.
				return acc + (new_val * active_mask)
			
			new_acc_stats = jax.tree.map(accumulate_leaf, acc_stats, info)

			# 4. Update Finished Mask
			next_finished_mask = jnp.logical_or(finished_mask, next_done)

			return (
				train_state,
				next_env_state,
				memories,
				memories_mask,
				memories_mask_idx,
				next_obsv,
				next_done,
				step_env_currentloop + 1,
				next_finished_mask,
				new_acc_reward,
				new_acc_length,
				new_acc_stats,
				rng,
			), _

		(final_carry), _ = jax.lax.scan(
			_env_step, init_runner_state, None, num_steps
		)

		# --- 5. Post-Processing ---
		final_raw_rewards = final_carry[9]
		final_raw_lengths = final_carry[10]
		final_stats = final_carry[11]
		finished_mask = final_carry[8]

		count_finished = finished_mask.sum()
		
		# Helper arrays where Unfinished = NaN (for min/max/median)
		rewards_for_stats = jnp.where(finished_mask, final_raw_rewards, jnp.nan)
		lengths_for_stats = jnp.where(finished_mask, final_raw_lengths, jnp.nan)

		# --- A. Basic Statistics ---
		def get_stats(data_array, name):
			# Safe aggregation handles NaNs automatically
			return {
				f"{name}": jnp.where(count_finished > 0, jnp.nanmean(data_array), -jnp.inf),
			}


		metrics = {}
		metrics.update(get_stats(rewards_for_stats, "mean_return"))
		metrics["mean_performance"] = metrics["mean_return"] / 226.0 * 100.0 
		metrics.update(get_stats(lengths_for_stats, "average_episode_length"))
		# --- C. Achievements ---
		for key, val in final_stats.items():
			if "Achievements" in key:
				skill_name_raw = key.split("/")[-1]
				valid_stats = jnp.where(finished_mask, val, 0.0)
				mean_stat = jnp.where(count_finished > 0, valid_stats.sum() / count_finished, 0.0)
				metrics[f"skill_{skill_name_raw}"] = mean_stat

		return metrics

	return evaluate


def main(config, rng, train_state=None, eval_embedding=None):
	# 1. Create the base environment
	if eval_embedding is not None:
		embedding_size = eval_embedding.shape[1]
		env = CraftaxAugObsTrain(
			condition_on_task=config.training.condition_on_task,
			conditioning_type="embedding",
			embedding_size=embedding_size,
			task_embeddings=eval_embedding,
		)
	else:
		env = CraftaxAugObsTrain()

	env_params = env.default_params.replace(
		max_timesteps=8192,
	)

	env = BatchEnvWrapper(env, num_envs=config.evaluation.num_envs)


	rng, _rng = jax.random.split(rng)
	evaluate_jit = jax.jit(make_evaluate(config, env, env_params))
	metrics = evaluate_jit(train_state, _rng)
	return metrics


if __name__ == "__main__":
	main()
