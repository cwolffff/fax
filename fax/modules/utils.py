from jax import numpy as jnp


def standardize(x: jnp.array, eps=1e-5) -> jnp.array:
    return (x - x.mean()) / (x.std() + eps)
