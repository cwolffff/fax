import jax
from jax import numpy as jnp
from jax.random import KeyArray

from .config import TransformerConfig
from ..basic import ElementwiseLinear
from ..basic import Linear
from ..core import Module
from ..utils import standardize


class TransformerHead(Module):

  nodes = [
    "query",
    "key",
    "value",
  ]

  def __init__(
      self,
      key: KeyArray,
      d_model: int,
      d_k: int,
  ):
    qkey, kkey, vkey = jax.random.split(key, num=3)
    self.query = Linear(
      key=qkey,
      in_features=d_model,
      out_features=d_k,
    )
    self.key = Linear(
      key=kkey,
      in_features=d_model,
      out_features=d_k,
    )
    self.value = Linear(
      key=vkey,
      in_features=d_model,
      out_features=d_model,
    )


class TransformerLayer(Module):

  nodes = [
    "norm_self_attn",
    "heads",
    "norm_ff",
    "ffn1",
    "ffn2",
  ]

  def __init__(
      self,
      key: KeyArray,
      n_heads: int,
      d_model: int,
      d_k: int,
      d_ff: int,
      tau: float,
  ):
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_k = d_k
    self.d_ff = d_ff
    self.tau = tau

    self.norm_self_attn = ElementwiseLinear(d_model)

    key, *subkeys = jax.random.split(key, num=n_heads + 1)
    self.heads = [
      TransformerHead(key=subkey, d_model=d_model, d_k=d_k)
      for subkey in subkeys
    ]

    self.norm_ff = ElementwiseLinear(d_model)

    subkey1, subkey2 = jax.random.split(key, num=2)
    self.ffn1 = Linear(subkey1, d_model, d_ff)
    self.ffn2 = Linear(subkey2, d_ff, d_model)
  
  def __call__(self, x: jnp.array, mask) -> jnp.array:

    # Layer-normalize
    t1 = jax.vmap(standardize)(x)
    t1 = self.norm_self_attn(t1)                           # L x Dm

    # Multi-head self-attention
    for head in self.heads:

      # Project into this head's query/key space
      query = head.query(t1)                               # L x Dk
      key = head.key(t1)                                   # L x Dk

      # Compute L x L attention matrix
      score = query @ key.T + mask                         # L x L
      attn = jax.nn.softmax(self.tau * score, axis=1)  # L x L

      value = head.value(t1)                               # L x Dm
      self_attn = attn @ value                             # L x Dm

      # Add this head's contribution
      x += self_attn                                       # L x Dm

    # Layer-normalize
    t2 = jax.vmap(standardize)(x)
    t2 = self.norm_ff(t2)                                  # L x Dm

    # Feedforward fully connected
    t2 = self.ffn1(t2)                                     # L x Dff
    t2 = jax.nn.relu(t2)
    t2 = self.ffn2(t2)                                     # L x Dm

    # Add this layer's contribution
    x += t2

    return x


class Transformer(Module):

  nodes = [
    "embeddings",
    "positional_encodings",
    "layers",
    "pre_output_norm",
    "output",
  ]

  def __init__(self, key: KeyArray, cfg: TransformerConfig):
    self.cfg = cfg

    key, subkey = jax.random.split(key)
    self.embeddings = jax.random.normal(subkey, shape=(cfg.vocab_size, cfg.d_model))
    self.positional_encodings = jnp.zeros((cfg.max_len, cfg.d_model))

    key, *subkeys = jax.random.split(key, num=cfg.n_layers+1)
    self.layers = [
      TransformerLayer(
        key=key,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_k=cfg.d_k,
        d_ff=cfg.d_ff,
        tau=cfg.tau,
      )
      for subkey in subkeys
    ]

    self.pre_output_norm = ElementwiseLinear(cfg.d_model)
    self.output = Linear(key, cfg.d_model, cfg.vocab_size)

  def __call__(self, x: jnp.ndarray):
    """
    cfg: Config, from transformer_init, holds hyperparameters
    nodes: Current transformer parameters, initialized in init
    x: 1D array of L integers, representing the input sequence
    output: L x n_vocab logits
    """

    L, = x.shape

    # Mask entries: 0 to attend, -Inf to ignore
    mask = jnp.log(jnp.tril(jnp.ones((L, L))))

    embeddings = self.cfg.lambda_e * self.embeddings[x, :]

    # Add (trainable) positional encodings
    latents = embeddings + self.cfg.lambda_pe * self.positional_encodings[:L, :]

    # Apply the transformer layers
    for layer in self.layers:
      latents += layer(latents, mask)

    # Layer-normalize
    latents = jax.vmap(standardize)(latents)
    latents = self.pre_output_norm(latents)

    # Linearly project to output dimension
    return self.output(latents)
  
  def generate(self, seq: jnp.ndarray, length: int = 20):
    for _ in range(length):
      output = self(seq)
      idx = jnp.argmax(output[-1])
      seq = jnp.append(seq, idx)
    return seq

