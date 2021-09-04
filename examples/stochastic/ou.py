import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

num_dims = 20
t = jnp.linspace(0.0, 20.0, num_dims + 1)[1:]
theta = 0.5
mu = 1.0
sigma = 0.5
sigmasq = jnp.square(sigma)

xo = 10.0
mean = xo*jnp.exp(-theta*t) + mu*(1.0 - jnp.exp(-theta*t))
Cov = sigmasq / (2*theta) * (jnp.exp(-theta*jnp.abs(t - t[..., jnp.newaxis])) - jnp.exp(-theta*jnp.abs(t + t[..., jnp.newaxis])))

dist = tfp.distributions.MultivariateNormalFullCovariance(mean, Cov)

def log_prob(x):
    lp = dist.log_prob(x)
    return lp

def sample(shape, rng):
    x = dist.sample(shape, seed=rng)
    return x
