import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
jtf = tfp.tf2jax


dist = tfd.MultivariateStudentTLinearOperator(
    df=15.0,
    loc=jnp.zeros(2),
    scale=jtf.linalg.LinearOperatorLowerTriangular(jnp.eye(2))
)

def log_prob(x):
    lp = dist.log_prob(x)
    return lp

def sample(shape, rng):
    x = dist.sample(shape, seed=rng)
    return x
