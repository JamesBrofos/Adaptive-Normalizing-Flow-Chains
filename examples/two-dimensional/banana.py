import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
tfb = tfp.bijectors

class Banana(tfb.Bijector):
    """Creates a bijector which, when applied to a correlated Gaussian distribution
    produces a distribution that is shaped like a banana. Code adapted from [1].

    [1] https://tinyurl.com/u4k98wpa

    """
    def __init__(self, name="banana"):
        super(Banana, self).__init__(inverse_min_event_ndims=1,
                                     is_constant_jacobian=True,
                                     name=name)

    def _forward(self, x):
        y_0 = x[..., 0:1]
        y_1 = x[..., 1:2] - y_0**2 - 1
        y_tail = x[..., 2:-1]
        y = jnp.concatenate([y_0, y_1, y_tail], axis=-1)
        return y

    def _inverse(self, y):
        x_0 = y[..., 0:1]
        x_1 = y[..., 1:2] + x_0**2 + 1
        x_tail = y[..., 2:-1]
        x = jnp.concatenate([x_0, x_1, x_tail], axis=-1)
        return x

    def _inverse_log_det_jacobian(self, y):
        return jnp.zeros(shape=())

Sigma = jnp.array([[1.00, 0.95], [0.95, 1.00]])
L = jnp.linalg.cholesky(Sigma)
dist = Banana()(tfd.MultivariateNormalTriL(scale_tril=L))

def log_prob(x):
    lp = dist.log_prob(x)
    return lp

def sample(shape, rng):
    x = dist.sample(shape, seed=rng)
    return x

