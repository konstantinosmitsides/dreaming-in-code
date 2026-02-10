import functools
from collections.abc import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax import struct
from dicode.transformer.transformerXL import Transformer


class ScannedRNN(nn.Module):
	@functools.partial(
		nn.scan,
		variable_broadcast="params",
		in_axes=0,
		out_axes=0,
		split_rngs={"params": False},
	)
	@nn.compact
	def __call__(self, carry, x):
		"""Applies the module."""
		rnn_state = carry
		ins, resets = x
		rnn_state = jnp.where(
			resets[:, np.newaxis],
			self.initialize_carry(ins.shape[0], ins.shape[1]),
			rnn_state,
		)
		new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
		return new_rnn_state, y

	@staticmethod
	def initialize_carry(batch_size, hidden_size):
		# Use a dummy key since the default state init fn is just zeros.
		cell = nn.GRUCell(features=hidden_size)
		return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
	action_dim: Sequence[int]
	config: dict

	@nn.compact
	def __call__(self, hidden, x):
		obs, dones = x
		embedding = nn.Dense(
			self.config.layer_size,
			kernel_init=orthogonal(np.sqrt(2)),
			bias_init=constant(0.0),
		)(obs)
		embedding = nn.relu(embedding)

		rnn_in = (embedding, dones)
		hidden, embedding = ScannedRNN()(hidden, rnn_in)

		actor_mean = nn.Dense(
			self.config.layer_size,
			kernel_init=orthogonal(2),
			bias_init=constant(0.0),
		)(embedding)
		actor_mean = nn.relu(actor_mean)
		actor_mean = nn.Dense(
			self.config.layer_size,
			kernel_init=orthogonal(2),
			bias_init=constant(0.0),
		)(actor_mean)
		actor_mean = nn.relu(actor_mean)
		actor_mean = nn.Dense(
			self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
		)(actor_mean)

		pi = distrax.Categorical(logits=actor_mean)

		critic = nn.Dense(
			self.config.layer_size,
			kernel_init=orthogonal(2),
			bias_init=constant(0.0),
		)(embedding)
		critic = nn.relu(critic)
		critic = nn.Dense(
			self.config.layer_size,
			kernel_init=orthogonal(2),
			bias_init=constant(0.0),
		)(critic)
		critic = nn.relu(critic)
		critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

		return hidden, pi, jnp.squeeze(critic, axis=-1)


# --- TRANSITION TUPLE ---
@struct.dataclass
class Transition:
	done: jnp.ndarray
	action: jnp.ndarray
	value: jnp.ndarray
	reward: jnp.ndarray
	log_prob: jnp.ndarray
	memories_mask: jnp.ndarray
	memories_indices: jnp.ndarray
	obs: jnp.ndarray
	info: jnp.ndarray


# --- TRANSFORMER ---
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
