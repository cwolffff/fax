import jax
from jax import numpy as jnp
from jax.random import KeyArray

from .core import Module


class Linear(Module):
  """A linear function, computing Ax + b.

  Initialized with uniform random weights and zero bias.
  """

  nodes = ["weight", "bias"]

  def __init__(
      self,
      key: KeyArray,
      in_features: int,
      out_features: int,
      with_bias: bool = True,
  ):
    self.in_features = in_features
    self.out_features = out_features
    self.with_bias = with_bias

    maxval = 1 / in_features ** 0.5
    minval = -maxval
    self.weight = jax.random.uniform(
      key,
      shape=(in_features, out_features),
      minval=minval,
      maxval=maxval,
    )

    if with_bias:
      self.bias = jnp.zeros((out_features,))

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    out = jnp.dot(inputs, self.weight)

    if self.with_bias:
      bias = jnp.broadcast_to(self.bias, out.shape)
      out = out + bias

    return out


class ElementwiseLinear(Module):
  """
  An elementwise linear layer.
  
  Initialized with unit gain and zero bias.
  """

  nodes = ["gain", "bias"]

  def __init__(self, shape):
    self.gain = jnp.ones(shape)
    self.bias = jnp.zeros(shape)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    # TODO: check if there's an alternative to broadcasting here
    gain = jnp.broadcast_to(self.gain, x.shape)
    out = gain * x
    bias = jnp.broadcast_to(self.bias, out.shape)
    out += bias
    return out
