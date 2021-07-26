import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

dist = tfd.JointDistributionSequential([
    tfd.Normal(loc=0.0, scale=1.0),
    lambda x: tfd.Normal(loc=0.0, scale=jnp.exp(0.5*x))
])

def log_prob(x):
    lp = dist.log_prob([x[..., 0], x[..., 1]])
    return lp

def sample(shape, rng):
    x = dist.sample(shape, seed=rng)
    x = jnp.stack(x, axis=-1)
    return x
