"""Sampling For Learnability (SFL) on Craftax, with PPO-GTrXL.

Derived from NCC-UED (https://github.com/nmonette/NCC-UED, Apache-2.0) — see baselines/NOTICE.
Modified for the paper: the feed-forward policy is replaced with the Gated Transformer-XL shared
with DiCode (`dicode.transformer.transformerXL`), and the PPO hyperparameters are set to the
paper's Table 6. The argparse defaults below are the values the paper's runs used.

Run from the repository root:

    python -m baselines.sfl --seed <SEED>
"""

import json
import os
import time
import datetime
from enum import IntEnum
from typing import Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG
from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from craftax.craftax.renderer import render_craftax_pixels as render_pixels
from craftax.craftax_classic.renderer import render_craftax_pixels as render_pixels_classic
from craftax.craftax.world_gen.world_gen import generate_world as generate_world_craftax
from craftax.craftax_classic.world_gen import generate_world as generate_world_classic
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
from flax import core, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState

import wandb
from jaxued.environments.underspecified_env import (EnvParams, EnvState,
                                                    Observation,
                                                    UnderspecifiedEnv)
from jaxued.level_sampler import LevelSampler as BaseLevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper
from baselines.mutators import (make_mutator_craftax_mutate_angles,
                       make_mutator_craftax_swap,
                       make_mutator_craftax_swap_restricted)

from functools import partial
from baselines.craftax_wrappers import CraftaxLoggerGymnaxWrapper, LogWrapper

# The GTrXL shared with DiCode — this is what makes the "same architecture across all methods"
# claim in the paper literal rather than a coincidence of two copies.
from dicode.transformer.transformerXL import Transformer




@struct.dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    memories_mask: jnp.ndarray  # Moved up to match instantiation
    memories_indices: jnp.ndarray  # Moved up to match instantiation
    obs: jnp.ndarray
    info: jnp.ndarray


# --- 2. Transformer Network Class ---
class ActorCriticTransformer(nn.Module):
    action_dim: int
    activation: str
    hidden_layers: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating: bool = False
    gating_bias: float = 0.0

    def setup(self):
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
        """Used during environment rollout (single timestep). Returns memory."""
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
        """Used during training: a window of observation is sent. Does NOT return memory."""
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


# --- Helper Functions for Transformer Logic ---
indices_select = lambda x, y: x[y]
batch_indices_select = jax.vmap(indices_select)
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])

LAYER_WIDTH = 512

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_updates: int
    total_env_steps: int
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

class LevelSampler(BaseLevelSampler):

    def level_weights(self, sampler, *args,**kwargs):
        return sampler["scores"]
    
    def initialize(self, levels, level_extras):
        sampler = {
                "levels": levels,
                "scores": jnp.full(self.capacity, 1 / self.capacity, dtype=jnp.float32),
                "timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
                "size": self.capacity,
                "episode_count": 0,
        }
        if level_extras is not None:
            sampler["levels_extra"] = level_extras
        return sampler

# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values

def sample_trajectories(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_obs: Observation,
    init_env_state: EnvState,
    init_memories: chex.Array,
    init_mask: chex.Array,
    init_mask_idx: chex.Array,
    init_done: chex.Array,
    init_step_env_currentloop: int,
    num_envs: int,
    window_mem: int,
    num_heads: int,
    max_episode_length: int,
    gamma: float = 0.99,
    give_returns: bool = False,
    record_trajectory: bool = True
) -> Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, obs, env_state, disc_factor, returns, valid_mask, memories, memories_mask, memories_mask_idx, done, step_env_currentloop = carry
        # rng, rng_action, rng_step = jax.random.split(rng, 3)

        # pi, value = train_state.apply_fn(train_state.params, obs)
        # action = pi.sample(seed=rng_action)
        # log_prob = pi.log_prob(action)

        # next_obs, env_state, reward, done, info = jax.vmap(
        #     env.step, in_axes=(0, 0, 0, None)
        # )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        memories_mask_idx = jnp.where(
            done, window_mem, jnp.clip(memories_mask_idx - 1, 0, window_mem)
        )
        memories_mask = jnp.where(
            done[:, None, None, None],
            jnp.zeros(
                (num_envs, num_heads, 1, window_mem + 1),
                dtype=jnp.bool_,
            ),
            memories_mask,
        )

        # 2. Update memories mask with the potential additional step
        memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, window_mem + 1)
        memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
            num_heads, 1
        )
        memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

        # 3. Select Action
        rng, _rng = jax.random.split(rng)
        pi, value, memories_out = train_state.apply_fn(train_state.params, memories, obs, memories_mask, method="model_forward_eval")
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # 4. Update Cache: Roll memory and add new output
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

        # 5. Step Env
        rng, _rng = jax.random.split(rng)
        # obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        # obsv, env_state, reward, done, info = jax.vmap(
        #     env.step, in_axes=(0, 0, 0, None)
        # )(jax.random.split(_rng, num_envs), env_state, action, env_params)

        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(_rng, num_envs), env_state, action, env_params)

        # 6. Compute memory indices for training
        memory_indices = jnp.arange(0, window_mem)[
            None, :
        ] + step_env_currentloop * jnp.ones((num_envs, 1), dtype=jnp.int32)
    
        transition = Transition(
            done,
            action,
            value,
            reward,
            log_prob,
            memories_mask.squeeze(),
            memory_indices,
            obs,
            info,
        )

        valid_mask *= ~done
        returns += disc_factor * reward * valid_mask
        disc_factor *= gamma

        carry = (rng, train_state, obsv, env_state, disc_factor, returns, valid_mask, memories, memories_mask, memories_mask_idx, done, step_env_currentloop)

        if record_trajectory:
            return carry, (transition, memories_out)
        else:
            return carry, None

    scan_out = jax.lax.scan(
        sample_step,
        (rng, train_state, init_obs, init_env_state, 1.0, jnp.zeros(num_envs), jnp.ones(num_envs), init_memories, init_mask, init_mask_idx, init_done, init_step_env_currentloop),
        None,
        length=max_episode_length,
    )

    (rng, train_state, last_obs, last_state, _, returns, _, last_memories, last_mask, last_mask_idx, last_done, last_step_env_currentloop), scan_stack = scan_out

    if record_trajectory:
        traj_batch, memories_batch = scan_stack
    else:
        traj_batch, memories_batch = None, None
    
    _, last_value, _ = train_state.apply_fn(train_state.params, last_memories, last_obs, last_mask, method="model_forward_eval")
    
    if not give_returns:
        return (rng, train_state, last_obs, last_state, last_value, last_memories, last_mask, last_mask_idx, last_done, last_step_env_currentloop), (traj_batch, memories_batch)
    else:
        return (rng, train_state, last_obs, last_state, last_value, returns, last_memories, last_mask, last_mask_idx, last_done, last_step_env_currentloop), (traj_batch, memories_batch)

def update_actor_critic(
    rng: chex.PRNGKey,
    train_state: TrainState,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    window_grad: int,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            traj_batch, memories_batch, advantages, targets = minibatch

            def loss_fn(params, traj_batch, memories_batch, gae, targets):
                # --- TRANSFORMER SPECIFIC: MEMORY BATCHING ---
                # Construct memory batch from indices
                memories_batch = batch_indices_select(
                    memories_batch, traj_batch.memories_indices[:, :: window_grad]
                )
                memories_batch = batchify(memories_batch)

                # Create Mask for Window Grad
                memories_mask = traj_batch.memories_mask.reshape(
                    (
                        -1,
                        window_grad,
                    )
                    + traj_batch.memories_mask.shape[2:]
                )
                memories_mask = jnp.swapaxes(memories_mask, 1, 2)
                # Concatenate with 0s
                memories_mask = jnp.concatenate(
                    (
                        memories_mask,
                        jnp.zeros(
                            memories_mask.shape[:-1] + (window_grad - 1,),
                            dtype=jnp.bool_,
                        ),
                    ),
                    axis=-1,
                )
                # Roll
                memories_mask = roll_vmap(
                    memories_mask, jnp.arange(0, window_grad), -1
                )

                # Reshape Obs and Batch
                obs = traj_batch.obs.reshape(
                    (
                        -1,
                        window_grad,
                    )
                    + traj_batch.obs.shape[2:]
                )
                traj_batch_r, targets_r, gae_r = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, window_grad) + x.shape[2:]),
                    (traj_batch, targets, gae),
                )

                pi, value = train_state.apply_fn(
                    params,
                    memories_batch,
                    obs,
                    memories_mask,
                    method="model_forward_train",
                )
                log_prob = pi.log_prob(traj_batch_r.action)

                # Value Loss
                value_pred_clipped = traj_batch_r.value + (value - traj_batch_r.value).clip(
                    -clip_eps, clip_eps
                )
                value_losses = jnp.square(value - targets_r)
                value_losses_clipped = jnp.square(value_pred_clipped - targets_r)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                # Actor Loss
                ratio = jnp.exp(log_prob - traj_batch_r.log_prob)
                gae_r = (gae_r - gae_r.mean()) / (gae_r.std() + 1e-8)
                loss_actor1 = ratio * gae_r
                loss_actor2 = (
                    jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_r
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                entropy = pi.entropy().mean()
                total_loss = (
                    loss_actor + critic_coeff * value_loss - entropy_coeff * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params, traj_batch, memories_batch, advantages, targets)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)

            grad_norm = jnp.linalg.norm(jnp.concatenate(jax.tree_util.tree_map(lambda x: x.flatten(), jax.tree_util.tree_flatten(grads)[0])))
            return train_state, (loss, grad_norm)

        rng, train_state = carry
        rng, _rng = jax.random.split(rng)

        permutation = jax.random.permutation(_rng, num_envs)
        batch_transposed = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch_transposed
        )
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [n_minibatch, -1] + list(x.shape[1:])),
            shuffled_batch,
        )
        train_state, (losses, grads) = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), (losses, grads)


    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

def sample_trajectories_and_learn(
    env: UnderspecifiedEnv, 
    env_params: EnvParams, 
    config: dict,
    rng: chex.PRNGKey, 
    train_state: TrainState, 
    init_obs: Observation, 
    init_env_state: EnvState, 
    init_memories: chex.Array, 
    init_memories_mask: chex.Array, 
    init_memories_mask_idx: chex.Array, 
    init_done: chex.Array,
    init_step_env_currentloop: int,
    update_grad: bool=True) -> Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict, chex.Array, chex.Array, chex.ArrayTree, chex.Array]]:
    """This function loops the following:
        - rollout for config['num_steps']
        - learn / update policy
    
    And it loops it for config['outer_rollout_steps'].
    What is returns is a new carry (rng, train_state, init_obs, init_env_state), and concatenated rollouts. The shape of the rollouts are config['num_steps'] * config['outer_rollout_steps']. In other words, the trajectories returned by this function are the same as if we ran rollouts for config['num_steps'] * config['outer_rollout_steps'] steps, but the agent does perform PPO updates in between.

    Args:
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        config (dict): 
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_obs (Observation): 
        init_env_state (EnvState): 
        update_grad (bool, optional): Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict, chex.Array, chex.Array, chex.ArrayTree, chex.Array]]: This returns a tuple:
        (
            (rng, train_state, init_obs, init_env_state),
            (obs, actions, rewards, dones, log_probs, values, info, advantages, targets, losses, grads)
        )
    """
    

    def single_step(carry, _):
        current_loop_reset = 0

        rng, train_state, init_obs, init_env_state, init_memories, init_memories_mask, init_memories_mask_idx, init_done, init_step_env_currentloop = carry
        (
            (rng, train_state, last_obs, last_env_state, last_value, last_memories, last_mask, last_mask_idx, last_done, last_step_env_currentloop),
            (traj_batch, memories_batch),
        ) = sample_trajectories(
            rng,
            env,
            env_params,
            train_state,
            init_obs,
            init_env_state,
            init_memories,
            init_memories_mask,
            init_memories_mask_idx,
            init_done,
            current_loop_reset,
            config["num_train_envs"],
            config["window_mem"],
            config["num_heads"],
            config["num_steps"],
        )
        advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, traj_batch.value, traj_batch.reward, traj_batch.done)

        memories_batch = jnp.concatenate(
            [jnp.swapaxes(init_memories, 0, 1), memories_batch], axis=0
        )
        
        # Update the policy using trajectories collected from replay levels
        (rng, train_state), (losses, grads) = update_actor_critic(
            rng,
            train_state,
            (traj_batch, memories_batch, advantages, targets),
            config["num_train_envs"],
            config["num_steps"],
            config["num_minibatches"],
            config["epoch_ppo"],
            config["clip_eps"],
            config["entropy_coeff"],
            config["critic_coeff"],
            config["window_grad"],
            update_grad=update_grad,
        )
        new_carry = (rng, train_state, last_obs, last_env_state, last_memories, last_mask, last_mask_idx, last_done, last_step_env_currentloop)
        return new_carry, (traj_batch.obs, traj_batch.action, traj_batch.reward, traj_batch.done, traj_batch.log_prob, traj_batch.value, traj_batch.info, advantages, targets, losses, grads)

    
    carry = (rng, train_state, init_obs, init_env_state, init_memories, init_memories_mask, init_memories_mask_idx, init_done, init_step_env_currentloop)
    new_carry, all_rollouts = jax.lax.scan(single_step, carry, None, length=config['outer_rollout_steps'])

    all_rollouts = jax.tree_util.tree_map(lambda x: jnp.concatenate(x, axis=0), all_rollouts)
    return new_carry, all_rollouts

def evaluate(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_obs: Observation,
    init_env_state: EnvState,
    init_memories: chex.Array,
    init_memories_mask: chex.Array,
    init_memories_mask_idx: chex.Array,
    max_episode_length: int,
    num_envs: int,
    window_mem: int,
    num_heads: int,
    keep_states=True
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the model on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int): 

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]
    
    def step(carry, _):
        rng, obs, state, done, mask, episode_length, memories, memories_mask, memories_mask_idx = carry

        memories_mask_idx = jnp.where(
            done, window_mem, jnp.clip(memories_mask_idx - 1, 0, window_mem)
        )
        memories_mask = jnp.where(
            done[:, None, None, None],
            jnp.zeros(
                (num_envs, num_heads, 1, window_mem + 1),
                dtype=jnp.bool_,
            ),
            memories_mask,
        )

        # 2. Update memories mask with the potential additional step
        memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, window_mem + 1)
        memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
            num_heads, 1
        )
        memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

        # 3. Select Action
        rng, _rng = jax.random.split(rng)
        pi, value, memories_out = train_state.apply_fn(train_state.params, memories, obs, memories_mask, method="model_forward_eval")
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # 4. Update Cache: Roll memory and add new output
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

        # 5. Step Env
        rng, _rng = jax.random.split(rng)
        # obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        obsv, next_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(_rng, num_envs), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        if keep_states:
            return (rng, obsv, next_state, done, next_mask, episode_length, memories, memories_mask, memories_mask_idx), (state, reward)
        else:
            return (rng, obsv, next_state, done, next_mask, episode_length, memories, memories_mask, memories_mask_idx), (None, reward)
    
    (_, _, _, _, _, episode_lengths, _, _, _), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
            init_memories,
            init_memories_mask,
            init_memories_mask_idx,
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/run_name/seed/config.json`

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    run_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    rl_checkpoint_dir = os.path.join(run_dir, "rl_checkpoints")
    os.makedirs(rl_checkpoint_dir, exist_ok=True)
    
    # save the config
    with open(os.path.join(run_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config.as_dict(), indent=True))
    
    checkpoint_manager = ocp.CheckpointManager(
        rl_checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
        )
    )
    return checkpoint_manager
#endregion

def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.
    
    Args:
        train_state (TrainState): 
        level_sampler (LevelSampler): 

    Returns:
        dict: 
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)

    dist = train_state.sampler["scores"]
    return {
        "log":{
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
            # "level_sampler/adv_entropy": -jnp.log(dist + 1e-6).T @ dist,
        },
        "info": {}
    }

def compute_score(config: dict, dones: chex.Array, values: chex.Array, max_returns: chex.Array, advantages: chex.Array) -> chex.Array:
    # Computes the score for each level
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")

def main(config=None, project="JAXUED_TEST"):
    tags = ["SFL-TR"]
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    # else:
    #     tags.append("PLR")
    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags, entity=config["wandb_entity"])
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("total_env_steps")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"]
        env_steps_delta = config["eval_freq"] * config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "total_env_steps": stats["total_env_steps"],
            "num_env_steps": env_steps,
            "sps": env_steps_delta / stats['time_delta'],
        }
        
        # evaluation performance
        returns     = stats["eval_returns"]
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        losses = stats["losses"]

        log_dict.update({
            "total_loss": losses[0][-1],
            "value_loss": losses[1][0][-1],
            "policy_loss": losses[1][1][-1],
            "entropy_loss": losses[1][2][-1],
        })

        # level sampler
        log_dict.update(train_state_info["log"])
        
        wandb.log(log_dict)


    def sample_random_level(rng):
        if config['accel_mutation'] == 'noise':
            rng, _rng1, _rng2, _rng3, _rng4 = jax.random.split(rng, 5)
            larger_res = (DEFAULT_STATICS.map_size[0] // 4, DEFAULT_STATICS.map_size[1] // 4)
            small_res = (DEFAULT_STATICS.map_size[0] // 16, DEFAULT_STATICS.map_size[1] // 16)
            x_res = (DEFAULT_STATICS.map_size[0] // 8, DEFAULT_STATICS.map_size[1] // 2)
            fractal_noise_angles = (jax.random.uniform(_rng1, (small_res[0] + 1, small_res[1] + 1)), 
                                    jax.random.uniform(_rng2, (small_res[0] + 1, small_res[1] + 1)), 
                                    jax.random.uniform(_rng3, (x_res[0] + 1, x_res[1] + 1)), 
                                    jax.random.uniform(_rng4, (larger_res[0] + 1, larger_res[1] + 1)))
            params_to_use = env.default_params.replace(fractal_noise_angles=fractal_noise_angles)
            return generate_world(rng, params_to_use, DEFAULT_STATICS).replace(fractal_noise_angles=fractal_noise_angles)
        else:
            return generate_world(rng, env.default_params, DEFAULT_STATICS)

    
    # Setup the environment. 
    # TODO: Add support for Pixels
    if 'Pixels' in config['env_name']:  raise ValueError("Pixel-environments are not supported yet.") 
    is_classic = False
    if config['env_name'] == 'Craftax-Classic-Symbolic-v1':
        ENV_CLASS = CraftaxClassicSymbolicEnv
        generate_world = generate_world_classic
        render_craftax_pixels = render_pixels_classic
        is_classic = True
    elif config['env_name'] == 'Craftax-Classic-Pixels-v1':
        ENV_CLASS = CraftaxClassicPixelsEnv
        generate_world = generate_world_classic
        render_craftax_pixels = render_pixels_classic
        is_classic = True
    elif config['env_name'] == 'Craftax-Symbolic-v1':
        # The paper's SFL runs used MiniCraftax here — a CraftaxSymbolicEnv subclass that swaps in
        # parameterized versions of change_floor/spawn_mobs/update_mobs/update_plants/
        # update_player_intrinsics. At its default EnvParams every extension is neutral, and the two
        # were verified to produce bit-identical observations, rewards and terminations over
        # 3 seeds x 500 steps. CraftaxSymbolicEnv is used directly here so that SFL, PLR and DR
        # visibly share one environment.
        ENV_CLASS = CraftaxSymbolicEnv
        generate_world = generate_world_craftax
        render_craftax_pixels = render_pixels
    elif config['env_name'] == 'Craftax-Pixels-v1':
        ENV_CLASS = CraftaxPixelsEnv
        generate_world = generate_world_craftax
        render_craftax_pixels = render_pixels
    else:
        raise ValueError(f"Unknown environment: {config['env_name']}")
    
    DEFAULT_STATICS = ENV_CLASS.default_static_params()
    default_env = ENV_CLASS(DEFAULT_STATICS)
    env = LogWrapper(default_env)
    env = AutoReplayWrapper(env)
    eval_env = env
    # env_params = env.default_params
    env_params = env.default_params.replace(
        max_timesteps=8000,
    )
    obs_dim = default_env.observation_space(env_params).shape[0]
    # What mutator do we use?
    if config['accel_mutation'] == 'noise':
        mutate_level = make_mutator_craftax_mutate_angles(generate_world, DEFAULT_STATICS, env.default_params)
    elif config['accel_mutation'] == 'swap_restricted':
        mutate_level = make_mutator_craftax_swap_restricted(DEFAULT_STATICS, one_should_be_middle=True, is_craftax_classic=is_classic)
    elif config['accel_mutation'] == 'swap':
        mutate_level = make_mutator_craftax_swap(DEFAULT_STATICS, only_middle=True, is_craftax_classic=is_classic)
    else:
        raise ValueError(f"Unknown mutation type: {config['accel_mutation']}")
        
    # And the level sampler    
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config['topk_k']},
        duplicate_check=config['buffer_duplicate_check'],
    )

    @partial(jax.jit, static_argnums=(2, ))
    def learnability_fn(rng, levels, num_envs, train_state):

        init_memories = jnp.zeros(
            (num_envs, config["window_mem"], config["num_layers"], config["embed_size"])
        )
        init_memories_mask = jnp.zeros(
            (num_envs, config["num_heads"], 1, config["window_mem"] + 1), dtype=jnp.bool_
        )
        init_memories_mask_idx = jnp.zeros((num_envs,), dtype=jnp.int32) + (config["window_mem"] + 1)
        init_done = jnp.zeros((num_envs,), dtype=jnp.bool_)
        init_step_env_currentloop = 0
        def rollout_fn(rng):

            # Get the scores of the levels
            rng, _rng = jax.random.split(rng)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, num_envs), levels, env_params)
            # Rollout
            (
                (rng, _, last_obs, last_state, last_value, disc_return, last_memories, last_mask, last_mask_idx, last_done, last_step_env_currentloop),
                (_, _),
            ) = sample_trajectories(
                rng,
                env,
                env_params,
                train_state,
                init_obs,
                init_env_state,
                init_memories,
                init_memories_mask,
                init_memories_mask_idx,
                init_done,
                init_step_env_currentloop,
                num_envs,
                config["window_mem"],
                config["num_heads"],
                config["learn_num_steps"],
                1.0,
                give_returns=True,
                record_trajectory=False
            )
            return disc_return
        
        rng, _rng = jax.random.split(rng)
        num_samples = 5
        returns = jax.vmap(rollout_fn)(jax.random.split(_rng, num_samples))
        
        # fit gaussian to mean episode returns
        
        def gaussian_pdf(x, mean, var):
            return (1 / (jnp.sqrt(2 * jnp.pi) * var)) * jnp.exp(-0.5 * ((x - mean) / var) ** 2)
        
        mean_returns = jnp.mean(returns, axis=0)
        mu = mean_returns.mean()
        sigma_2 = mean_returns.var()
        
        pdf_values = gaussian_pdf(mean_returns, mu, sigma_2)
        
        scores = jnp.sqrt(returns.var(axis=0)) / jnp.sqrt(num_samples) * pdf_values + 1e-4 * mean_returns
        
        return scores, returns.max(axis=0)

    # def get_learnability_set(rng, train_state):

    #     curation_steps = config["num_set_batches"] * config["level_buffer_capacity"] * 5 * 500

    #     def batch(_, rng):

    #         # Sample new levels
    #         rng, _rng = jax.random.split(rng)
    #         new_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["level_buffer_capacity"]))

    #         rng, _rng = jax.random.split(rng)
    #         new_level_scores, max_returns = learnability_fn(_rng, new_levels, config["level_buffer_capacity"], train_state)

    #         return None, (new_level_scores, new_levels)

    #     new_level_scores, new_levels = jax.tree_util.tree_map(
    #         lambda x: x.reshape(-1, *x.shape[2:]), jax.lax.scan(batch, None, xs = jax.random.split(rng, config["num_set_batches"]))[1]
    #     )

    #     idxs = jnp.flipud(jnp.argsort(new_level_scores))[:config["level_buffer_capacity"]]

    #     new_levels = jax.tree_util.tree_map(
    #         lambda x: x[idxs], new_levels
    #     )

    #     sampler = train_state.sampler
    #     sampler["levels"] = new_levels

    #     return train_state.replace(
    #         sampler = {**train_state.sampler, "levels": new_levels},
    #         total_env_steps=train_state.total_env_steps + curation_steps)

    def get_learnability_set(rng, train_state):
        # We need to process levels in chunks to avoid OOM
        # 4000 levels is too many to score at once. 
        # 250 levels * 5 samples = 1250 concurrent envs (Much safer than 20,000)

        curation_steps = config["num_set_batches"] * config["level_buffer_capacity"] * 5 * config["learn_num_steps"]
        SCORE_BATCH_SIZE = 250 
        
        # Ensure the capacity is divisible by the batch size for simplicity
        assert config["level_buffer_capacity"] % SCORE_BATCH_SIZE == 0, "Capacity must be divisible by batch size"
        num_scoring_batches = config["level_buffer_capacity"] // SCORE_BATCH_SIZE

        def batch(_, rng):
            # 1. Sample new levels (Total: 4000)
            rng, _rng = jax.random.split(rng)
            new_levels_total = jax.vmap(sample_random_level)(jax.random.split(_rng, config["level_buffer_capacity"]))
            
            # 2. Reshape levels into batches: (num_batches, batch_size, ...)
            # new_levels_total is a tuple (maps, params). We tree_map to reshape both.
            def reshape_for_batch(x):
                return x.reshape((num_scoring_batches, SCORE_BATCH_SIZE) + x.shape[1:])
            
            levels_batched = jax.tree_util.tree_map(reshape_for_batch, new_levels_total)

            # 3. Define the loop to score one batch at a time
            def score_single_batch(rng, levels_batch):
                rng, _rng = jax.random.split(rng)
                # Note: We pass SCORE_BATCH_SIZE here instead of the full capacity
                scores, max_rets = learnability_fn(_rng, levels_batch, SCORE_BATCH_SIZE, train_state)
                return rng, scores

            # 4. Run the loop (scan) over the batches
            rng, _rng_scan = jax.random.split(rng)
            _, batched_scores = jax.lax.scan(score_single_batch, _rng_scan, levels_batched)
            
            # 5. Flatten the scores back to (4000,)
            new_level_scores = batched_scores.reshape(-1)
            
            return None, (new_level_scores, new_levels_total)

        # Run the outer loop (Generating multiple sets of candidates)
        # We only return the last set, or you can collect them. 
        # The logic below matches your original: generate `num_set_batches` sets, pick the best from the concatenated pool.
        
        # NOTE: Your original logic collected ALL generated levels across `num_set_batches` loops 
        # and then picked the top 4000. We preserve that logic here.
        
        rng, (all_scores, all_levels) = jax.lax.scan(batch, None, xs=jax.random.split(rng, config["num_set_batches"]))
        
        # Reshape to flatten the outer 'num_set_batches' loop
        new_level_scores = all_scores.reshape(-1)
        new_levels = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), all_levels)

        # Sort and take top K
        idxs = jnp.flipud(jnp.argsort(new_level_scores))[:config["level_buffer_capacity"]]

        new_levels = jax.tree_util.tree_map(
            lambda x: x[idxs], new_levels
        )

        sampler = train_state.sampler
        sampler["levels"] = new_levels

        return train_state.replace(
            sampler = {**train_state.sampler, "levels": new_levels},
            total_env_steps=train_state.total_env_steps + curation_steps)


    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / (config["num_updates"] * config['outer_rollout_steps'])
            )
            return config["lr"] * frac
        # obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        rng, rng_level, rng_reset = jax.random.split(rng, 3)
        # init_map, init_params = sample_random_level(rng_level)
        # obs, _ = env.reset_to_level(rng_reset, init_map, init_params)

        init_obs = jnp.zeros((2, obs_dim))
        init_memory = jnp.zeros((2, config["window_mem"], config["num_layers"], config["embed_size"]))
        init_mask = jnp.zeros((2, config["num_heads"], 1, config["window_mem"] + 1), dtype=jnp.bool_)

        network = ActorCriticTransformer(
            action_dim=env.action_space(env_params).n,
            activation=config["activation"],
            hidden_layers=config["hidden_layers"],
            encoder_size=config["embed_size"],
            num_heads=config["num_heads"],
            qkv_features=config["qkv_features"],
            num_layers=config["num_layers"],
            gating=config["gating"],
            gating_bias=config["gating_bias"],
            )

        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, init_memory, init_obs, init_mask)


        tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate = linear_schedule, eps=1e-5)
            )
        rng, _rng = jax.random.split(rng)
        init_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["level_buffer_capacity"]))
        sampler = level_sampler.initialize(init_levels, {"max_return": jnp.full(config["level_buffer_capacity"], -jnp.inf)})
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            num_updates=0,
            total_env_steps=0,
            replay_last_level_batch=init_levels,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], t):


        init_memories = jnp.zeros(
            (config["num_train_envs"], config["window_mem"], config["num_layers"], config["embed_size"])
        )
        init_memories_mask = jnp.zeros(
            (config["num_train_envs"], config["num_heads"], 1, config["window_mem"] + 1), dtype=jnp.bool_
        )
        init_memories_mask_idx = jnp.zeros((config["num_train_envs"],), dtype=jnp.int32) + (config["window_mem"] + 1)
        init_done = jnp.zeros((config["num_train_envs"],), dtype=jnp.bool_)
        init_step_env_currentloop = 0
        
        rng, train_state = carry

        # Collect trajectories on replay levels
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        sampler, (level_inds, levels) = level_sampler.sample_replay_levels(train_state.sampler, rng_levels, config["num_train_envs"])

        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)

        (
            (rng, train_state, _, _, last_memories, last_mask, last_mask_idx, last_done, last_step_env_currentloop),
            (_, _, _, dones, _, _, info, _, _, losses, grads)
            ) = sample_trajectories_and_learn(
                env, 
                env_params, 
                config,
                rng, 
                train_state, 
                init_obs, 
                init_env_state, 
                init_memories,
                init_memories_mask,
                init_memories_mask_idx,
                init_done,
                init_step_env_currentloop,
                update_grad=True
            )

        steps_taken = config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"]

        train_state = train_state.replace(
            num_updates=train_state.num_updates + 1,
            total_env_steps=train_state.total_env_steps + steps_taken 
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "achievements": (info["achievements"] * dones[..., None]).sum(axis=0).sum(axis=0) / dones.sum(),
            "achievement_count": (info["achievement_count"] * dones).sum() / dones.sum(),
            "returned_episode_lengths": (info["returned_episode_lengths"] * dones).sum() / dones.sum(),
            "max_episode_length": info["returned_episode_lengths"].max(),
            "levels_played": init_env_state.env_state,
            "mean_returns": (info["returned_episode_returns"] * dones).sum() / dones.sum(),
            "grad_norms": grads.mean(),
        }
        
        return (rng, train_state), metrics
    
    
    def eval(rng: chex.PRNGKey, train_state: TrainState, keep_states=True):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        num_levels = config['n_eval_levels']
        levels = jax.vmap(generate_world, (0, None, None))(jax.random.split(jax.random.PRNGKey(10101), num_levels), env_params, DEFAULT_STATICS)
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        init_memories = jnp.zeros(
            (
                num_levels,
                config['window_mem'],
                config["num_layers"],
                config["embed_size"],
            )
        )
        init_memories_mask = jnp.zeros(
            (
                num_levels,
                config["num_heads"],
                1,
                config['window_mem'] + 1,
            ),
            dtype=jnp.bool_,
        )
        init_memories_mask_idx = jnp.zeros((num_levels,), dtype=jnp.int32) + (
            config['window_mem'] + 1
        )
        states, rewards, episode_lengths = evaluate(
            rng,
            eval_env,
            env_params,
            train_state,
            init_obs,
            init_env_state,
            init_memories,
            init_memories_mask,
            init_memories_mask_idx,
            config['num_eval_steps'],
            num_envs=num_levels,
            window_mem=config['window_mem'],
            num_heads=config['num_heads'],
            keep_states=keep_states
        )
        mask = jnp.arange(config['num_eval_steps'])[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, t):
        """
            This function runs the train_step for a certain number of iterations, and then evaluates the policy.
            It returns the updated train state, and a dictionary of metrics.
        """
        rng, train_state, xhat, prev_grad = runner_state
        rng, _rng = jax.random.split(rng)
        # train_state = get_learnability_set(_rng, train_state)

        # Only call get_learnability_set on even eval_steps
        train_state = jax.lax.cond(
            t.astype(jnp.int32) % 10 == 0,
            lambda x: get_learnability_set(x[0], x[1]),
            lambda x: x[1],
            (_rng, train_state)
        )

        runner_state = (rng, train_state)
        # Train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, jnp.arange(config["eval_freq"], dtype=float) + t)

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state, False)
        
        # Collect Metrics
        eval_returns = cum_rewards.mean(axis=0)
        episode_lengths = episode_lengths.mean(axis=0)      

        
        # just grab the first run
        # states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        
        # And one attempt
        # episode_lengths = episode_lengths[:1]
        
        metrics["update_count"] = train_state.num_updates
        metrics["total_env_steps"] = train_state.total_env_steps
        metrics["eval_returns"] = eval_returns
        metrics["eval_ep_lengths"]  = episode_lengths
        
        return (rng, train_state, xhat, prev_grad), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state, False)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths))
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval':
        return eval_checkpoint(config, ) # evaluate and exit early

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)

    # Set up y optimizer state        
    grad = jnp.zeros_like(train_state.sampler["scores"])
    rng, _rng = jax.random.split(rng)
    xhat = jnp.full_like(grad, 1 / len(grad))

    runner_state = (rng, train_state, xhat, grad)
    
    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)

    agent_updates_per_loop = config["eval_freq"] * config["outer_rollout_steps"]
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        # runner_state, metrics = train_and_eval_step(runner_state, eval_step * config["eval_freq"])
        runner_state, metrics = train_and_eval_step(runner_state, jnp.array(eval_step * config["eval_freq"], dtype=jnp.int32))
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["checkpoint_save_interval"] > 0:
            save_step_count = (eval_step + 1) * agent_updates_per_loop
            checkpoint_manager.save(save_step_count, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    return runner_state[1]

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="dicode-baselines")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Your W&B entity. Leave unset to use your default, or set WANDB_MODE=disabled.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    
    # === Train vs Eval ===
    parser.add_argument("--env_name", type=str, choices=['Craftax-Symbolic-v1', 'Craftax-Pixels-v1', 'Craftax-Classic-Symbolic-v1', 'Craftax-Classic-Pixels-v1'], default='Craftax-Symbolic-v1')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=0)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=100)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=2)
    parser.add_argument("--eval_num_attempts", type=int, default=1)
    group = parser.add_argument_group('Training params')
    # === PPO === 
    # group.add_argument("--lr", type=float, default=2e-4)
    group.add_argument("--max_grad_norm", type=float, default=1.0)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=120)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    # parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--outer_rollout_steps", type=int, default=64)
    # group.add_argument("--num_train_envs", type=int, default=1024)
    group.add_argument("--num_minibatches", type=int, default=8)
    group.add_argument("--gamma", type=float, default=0.999)
    group.add_argument("--epoch_ppo", type=int, default=4)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.8)
    group.add_argument("--entropy_coeff", type=float, default=0.002)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    # group.add_argument("--meta_lr", type=float, default=1e-2)
    # group.add_argument("--meta_trunc", type=float, default=1e-5)
    # group.add_argument("--meta_entr_coeff", type=float, default = 0.005)
    # group.add_argument("--meta_mix", type=float, default = 0.5)

    # === PPO-TR === 
    group.add_argument("--lr", type=float, default=2e-4)
    group.add_argument("--num_train_envs", type=int, default=1024)
    group.add_argument("--num_steps", type=int, default=128)
    group.add_argument("--qkv_features", type=int, default=256)
    group.add_argument("--embed_size", type=int, default=256)
    group.add_argument("--num_heads", type=int, default=8)
    group.add_argument("--num_layers", type=int, default=2)
    group.add_argument("--hidden_layers", type=int, default=256)
    group.add_argument("--window_mem", type=int, default=128)
    group.add_argument("--window_grad", type=int, default=64)
    group.add_argument("--gating", type=bool, default=True)
    group.add_argument("--gating_bias", type=float, default=2.0)
    group.add_argument("--activation", type=str, default="relu")

    # === PLR ===
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.5)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--topk_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--static_buffer", type=bool, default=False)
    group.add_argument("--num_set_batches", type=int, default=5)

    # === SFL ===
    parser.add_argument("--learn_num_steps", type=int, default=1500)
    # === ACCEL ===
    parser.add_argument("--use_accel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_edits", type=int, default=30)
    parser.add_argument("--accel_mutation", type=str, default="swap", choices=["swap", "swap_restricted", "noise"])

    # === Eval CONFIG ===
    parser.add_argument("--n_eval_levels", type=int, default=64)
    parser.add_argument("--num_eval_steps", type=int, default=2500)
    # === DR CONFIG ===
    
    config = vars(parser.parse_args())
    if config["run_name"] is None:
        config["run_name"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"])
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    wandb.login()
    main(config, project=config["project"])