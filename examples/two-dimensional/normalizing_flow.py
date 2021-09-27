from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfb = tfp.bijectors
tfd = tfp.distributions

class ShiftAndLogScale(hk.Module):
    """Parameterizes the shift and scale functions used in the RealNVP normalizing
    flow architecture.

    Args:
        num_hidden: Number of hidden units in each hidden layer.

    """
    def __init__(self, num_hidden: int):
        super().__init__()
        self._num_hidden = num_hidden

    def __call__(
            self,
            x: jnp.ndarray,
            input_depth: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = hk.Flatten()(x)
        x = hk.Linear(self._num_hidden)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_hidden)(x)
        x = jax.nn.relu(x)
        shift = hk.Linear(input_depth)(x)
        log_scale = hk.Linear(input_depth)(x)
        return shift, log_scale

def build_bijector(num_dims: int, num_hidden: int) -> tfb.Bijector:
    """Constructs a bijector object from the composition of RealNVP layers and
    permutation layers.

    Args:
        num_dims: The dimensionality of the input to the bijector.
        num_hidden: The number of hidden units used in the RealNVP bijectors.

    Returns:
        bij: The bijector composed of the concatenation of RealNVP and
            permutation bijectors.

    """
    num_masked = num_dims // 2
    bij = tfb.Chain([
        tfb.RealNVP(num_masked=num_masked,
                    shift_and_log_scale_fn=ShiftAndLogScale(num_hidden)),
        tfb.Permute(list(reversed(range(num_dims)))),
        tfb.RealNVP(num_masked=num_masked,
                    shift_and_log_scale_fn=ShiftAndLogScale(num_hidden)),
        tfb.Permute(list(reversed(range(num_dims)))),
        tfb.RealNVP(num_masked=num_masked,
                    shift_and_log_scale_fn=ShiftAndLogScale(num_hidden)),
        tfb.Permute(list(reversed(range(num_dims)))),
        tfb.RealNVP(num_masked=num_masked,
                    shift_and_log_scale_fn=ShiftAndLogScale(num_hidden)),
    ])
    return bij

def build_distribution(num_dims: int, num_hidden: int) -> tfd.TransformedDistribution:
    """Constructs the distribution associated to applying the bijector to a base
    distribution, which is a standard multivariate Gaussian.

    Args:
        num_dims: The dimensionality of the input to the bijector.
        num_hidden: The number of hidden units used in the RealNVP bijectors.

    Returns:
        dist: The distribution produced by applying the bijector to the base
            distribution.

    """
    dist = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(
            loc=jnp.zeros(num_dims),
            scale_diag=jnp.ones(num_dims)
        ),
        bijector=build_bijector(num_dims, num_hidden)
    )
    return dist

def NormalizingFlow(num_dims: int, num_hidden: int) -> Tuple[hk.Transformed, hk.Transformed]:
    """Constructs the sampling and log-density functions of the normalizing flow,
    which is composed of RealNVP bijectors and permutation layers.

    Args:
        num_dims: The dimensionality of the input to the bijector.
        num_hidden: The number of hidden units used in the RealNVP bijectors.

    Returns:
        sample: Transformed object with method to sample from the normalizing
            flow.
        log_prob: Transformed object with method to compute the log-density of
            the normalizing flow.

    """
    @hk.transform
    def sample(*args, **kwargs):
        return build_distribution(num_dims, num_hidden).sample(seed=hk.next_rng_key(), *args, **kwargs)

    @hk.without_apply_rng
    @hk.transform
    def log_prob(*args, **kwargs):
        return build_distribution(num_dims, num_hidden).log_prob(*args, **kwargs)

    return sample, log_prob
