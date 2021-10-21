from typing import Tuple

import autograd.numpy as np

import distributions


def brownian_bridge_target(num_times: int) -> Tuple[distributions.Distribution, int]:
    """Constructs the Brownian bridge target distribution with a specified
    dimensionality. This Brownian bridge differs from the typical depiction in
    that it uses a non-zero mean, which is sinusoidal.

    Args:
        num_times: The number of observations of the Brownian bridge.

    Returns:
        target: The Brownian bridge target distribution.
        num_dims: The dimensionality of the Brownian bridge distribution.

    """
    t = np.linspace(0.0, 1.0, num_times + 2)[1:-1]
    mean = np.sin(np.pi*t)
    cov = distributions.brownian_bridge_covariance(t)
    target = distributions.MultivariateNormal(mean, cov)
    num_dims = len(t)
    return target, num_dims

def multimodal_target() -> Tuple[distributions.MultivariateNormalMixture, int]:
    """Constructs a multimodal target distribution from two equally weighted
    Gaussian distributions. The target distribution is defined in the plane.

    Returns:
        target: The multimodal Gaussian target distribution.
        num_dims: The dimensionality of the multimodal target.

    """
    mus = 2.0*np.array([[-1.0, 1.0], [1.0, -1.0]])
    Id = np.eye(2)
    scale = 0.1
    sigmas = scale*np.array([Id, Id])
    probs = np.array([0.5, 0.5])
    target = distributions.MultivariateNormalMixture(mus, sigmas, probs)
    num_dims = 2
    return target, num_dims

def neal_funnel_target() -> Tuple[distributions.NealFunnel, int]:
    """Constructs the Neal funnel distribution.

    Returns:
        target: The Neal funnel target distribution.
        num_dims: The dimensionality of the multimodal target.

    """
    num_dims = 2
    target = distributions.NealFunnel(num_dims)
    return target, num_dims
