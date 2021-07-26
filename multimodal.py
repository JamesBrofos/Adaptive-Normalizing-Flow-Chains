import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

scale = 0.1
dist = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=jnp.ones(9)/9),
    components_distribution=tfd.MultivariateNormalDiag(
        loc=jnp.array([[-1.0,  1.0],
                       [ 0.0,  1.0],
                       [ 1.0,  1.0],
                       [-1.0,  0.0],
                       [ 0.0,  0.0],
                       [ 1.0,  0.0],
                       [-1.0, -1.0],
                       [ 0.0, -1.0],
                       [ 1.0, -1.0]]),
        scale_identity_multiplier=scale)
)

def log_prob(x):
    lp = dist.log_prob(x)
    return lp

def sample(shape, rng):
    x = dist.sample(shape, seed=rng)
    return x
