import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfb = tfp.bijectors
tfd = tfp.distributions

def build_multimodal_distribution():
    modes = jnp.array([[-1.0,  1.0],
                       [ 1.0, -1.0]])
    distr = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=jnp.ones(len(modes)) / len(modes)),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=modes,
            scale_identity_multiplier=0.1)
    )
    return distr


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

def build_banana_distribution():
    Sigma = jnp.array([[1.00, 0.95], [0.95, 1.00]])
    L = jnp.linalg.cholesky(Sigma)
    distr = Banana()(tfd.MultivariateNormalTriL(scale_tril=L))
    return distr

def build_neal_funnel_distribution():
    dist = tfd.JointDistributionSequential([
        tfd.Normal(loc=0.0, scale=1.0),
        lambda x: tfd.Normal(loc=0.0, scale=jnp.exp(0.5*x))
    ])
    class NealFunnel:
        def log_prob(self, x):
            lp = dist.log_prob([x[..., 0], x[..., 1]])
            return lp

        def sample(self, shape, seed):
            x = dist.sample(shape, seed=seed)
            x = jnp.stack(x, axis=-1)
            return x

    distr = NealFunnel()
    return distr


distributions = {
    'multimodal': build_multimodal_distribution(),
    'banana': build_banana_distribution(),
    'neal-funnel': build_neal_funnel_distribution()
}
