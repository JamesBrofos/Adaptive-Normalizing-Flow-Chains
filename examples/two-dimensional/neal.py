import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

dist = tfd.JointDistributionSequential([
    tfd.Normal(loc=0.0, scale=3.0),
    lambda x: tfd.MultivariateNormalDiag(
        loc=jnp.zeros(10),
        scale_identity_multiplier=jnp.exp(0.5*x)
    )
])

def log_prob(x):
    lp = dist.log_prob([x[..., 0], x[..., 1:]])
    return lp

def sample(shape, rng):
    x = dist.sample(shape, seed=rng)
    x = jnp.concatenate([x[0][..., jnp.newaxis], x[1]], axis=-1)
    return x
