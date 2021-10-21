import abc
import os
from typing import List, NamedTuple, Tuple

import autograd
import autograd.numpy as np
import autograd.scipy.linalg as spla
import autograd.scipy.special as spsp
import autograd.scipy.stats as spst
from autograd.builtins import tuple
from autograd.scipy.integrate import odeint


def norm_log_pdf(x: np.ndarray, mu: float, sigmasq: float) -> np.ndarray:
    """Computes the log-density function of the univariate normal distribution given
    its mean and variance.

    Args:
        x: Location at which to compute the log-density of the univariate normal
            distribution.
        mu: The mean of the univariate normal.
        sigmasq: The variance of the univariate normal.

    Returns:
        ld: The log-density of the univariate normal.

    """
    o = x - mu
    ld = -0.5*np.square(o) / sigmasq - 0.5*np.log(sigmasq) - 0.5*np.log(2*np.pi)
    return ld

def mvn_log_pdf(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray, log_det: float) -> np.ndarray:
    """Computes the log-density of a multivariate normal distribution given the
    log-determinant of the covariance matrix, the mean of the distribution, and
    the inverse of the covariance matrix.

    Args:
        x: Input at which to evaluate the log-density of the multivariate normal
            log-density.
        mu: The mean of the multivariate normal.
        sigma_inv: The inverse of the covariance matrix.
        log_det: The log-determinant of the covariance matrix.

    Returns:
        ld: The log-density of the multivariate normal distribution.

    """
    n = len(mu)
    o = x - mu
    l = (sigma_inv@o.T).T
    maha = np.sum(o*l, axis=-1)
    ld = -0.5*(n*np.log(2*np.pi) + log_det + maha)
    return ld

def mvn_sample(mu: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """Samples from a multivariate normal distribution given the mean and the
    Cholesky decomposition of the covariance matrix.

    Args:
        mu: The mean of the multivariate normal.
        chol: The Cholesky decomposition of the covariance matrix.

    Returns:
        x: A sample from the multivariate normal.

    """
    z = np.random.normal(size=mu.shape)
    x = chol@z + mu
    return x

def mvn_nll(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> float:
    """Computes the negative log-likelihood of the multivariate normal distribution
    given samples and values for the mean and covariance.

    Args:
        mu: The mean of the multivariate normal.
        sigma: The covariance matrix of the multivariate normal.
        x: Samples whose log-likelihood under a multivariate normal should be
            computed.

    Returns:
        nll: The negative log-likelihood of the observations given the mean and
            covariance matrix.

    """
    k = len(mu)
    Id = np.eye(k)
    chol = np.linalg.cholesky(sigma)
    chol_inv = spla.solve_triangular(chol, Id, lower=True)
    sigma_inv = chol_inv.T@chol_inv
    log_det = 2*np.sum(np.log(np.diag(chol)))
    nll = -np.mean(mvn_log_pdf(x, mu, sigma_inv, log_det))
    return nll


def mvn_kl(mu_a: np.ndarray, sigma_a: np.ndarray, mu_b: np.ndarray, sigma_b: np.ndarray) -> float:
    """Computes the KL divergence between two multivariate Gaussians given their
    means and covariances. We call the Gaussian with respect to whose
    expectation the KL divergence is computed the 'reference' distribution.

    Args:
        mu_a: Mean of the reference multivariate Gaussian.
        sigma_a: Covariance of the reference multivariate Gaussian.
        mu_b: Mean of the other multivariate Gaussian.
        sigma_b: Covariance of the other multivariate Gaussian.

    Returns:
        div: The KL divergence between two multivariate Gaussians.

    """
    o = mu_b - mu_a
    k = len(mu_a)
    chol_a = np.linalg.cholesky(sigma_a)
    chol_b = np.linalg.cholesky(sigma_b)
    chol_b_inv = spla.solve_triangular(chol_b, np.eye(k), lower=True)
    sigma_inv_b = chol_b_inv.T@chol_b_inv
    maha = o@sigma_inv_b@o
    log_det_a = 2*np.sum(np.log(np.diag(chol_a)))
    log_det_b = 2*np.sum(np.log(np.diag(chol_b)))
    log_det = log_det_b - log_det_a
    tr = np.trace(sigma_inv_b@sigma_a)
    div = 0.5*(tr + maha - k + log_det)
    return div

def mvn_ekl(mu_a: np.ndarray, sigma_a: np.ndarray, mu_b: np.ndarray, sigma_b: np.ndarray, num_samples: int) -> float:
    """Computes the empirical KL divergence between two multivariate Gaussian
    distributions. The KL divergence is the expected value of the difference in
    log-densities. In general, the expectation can be replaced by a Monte Carlo
    estimate, which is the practice here.

    Args:
        mu_a: Mean of the reference multivariate Gaussian.
        sigma_a: Covariance of the reference multivariate Gaussian.
        mu_b: Mean of the other multivariate Gaussian.
        sigma_b: Covariance of the other multivariate Gaussian.
        num_samples: The number of Monte Carlo samples to use to approximate the
            expectation.

    Returns:
        div: The estimated KL divergence between two multivariate Gaussians.

    """
    k = len(mu_a)
    Id = np.eye(k)

    chol_a = np.linalg.cholesky(sigma_a)
    chol_a_inv = spla.solve_triangular(chol_a, Id, lower=True)
    sigma_inv_a = chol_a_inv.T@chol_a_inv
    chol_b = np.linalg.cholesky(sigma_b)
    chol_b_inv = spla.solve_triangular(chol_b, Id, lower=True)
    sigma_inv_b = chol_b_inv.T@chol_b_inv

    log_det_a = 2*np.sum(np.log(np.diag(chol_a)))
    log_det_b = 2*np.sum(np.log(np.diag(chol_b)))

    x = mu_a + (chol_a@np.random.normal(size=(k, num_samples))).T
    div = np.mean(
        mvn_log_pdf(x, mu_a, sigma_inv_a, log_det_a) -
        mvn_log_pdf(x, mu_b, sigma_inv_b, log_det_b))
    return div

def brownian_bridge_covariance(t: np.ndarray) -> np.ndarray:
    """Computes the covariance matrix of the Brownian bridge. The Brownian bridge
    is a stochastic process such that the end points of the process are
    constrained to be equal. The Brownian bridge is a Gaussian process.

    Args:
        t: The time interval in which to compute the covariance.

    Returns:
        sigma: The covariance of the Brownian bridge.

    """
    s = t[..., np.newaxis]
    a = np.minimum(t, s)
    b = t*s
    sigma = a - b
    return sigma

class Distribution(abc.ABC):
    """Implements a basic distribution object. Distribution objects implement
    methods to compute their log-density at a given input and to draw i.i.d.
    samples from them. The log-density may be specified up to an additive
    constant.

    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def log_density(self, x: np.ndarray) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> np.ndarray:
        raise NotImplementedError()

class MultivariateNormal(Distribution):
    """The multivariate normal distribution is the multivariate generalization of
    the normal distribution. It is parameterized by its mean vector and its
    covariance matrix.

    Parameters:
        mu: The mean vector of the multivariate normal.
        sigma: The covariance matrix of the multivariate normal.

    """
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.chol = np.linalg.cholesky(sigma)
        self.log_det = 2*np.sum(np.log(np.diag(self.chol)))
        Id = np.eye(len(mu))
        self.sigma_inv = spla.cho_solve((self.chol, True), Id)

    def log_density(self, x: np.ndarray) -> float:
        ld = mvn_log_pdf(x, self.mu, self.sigma_inv, self.log_det)
        return ld

    def sample(self) -> np.ndarray:
        x = mvn_sample(self.mu, self.chol)
        return x

    def kl(self, other) -> float:
        o = other.mu - self.mu
        maha = o@other.sigma_inv@o
        log_det = other.log_det - self.log_det
        tr = np.trace(other.sigma_inv@self.sigma)
        div = 0.5*(tr + maha - len(self.mu) + log_det)
        return div

def node_sample_vector_field(z: np.ndarray, t: np.ndarray, params: Tuple[Tuple[np.ndarray]]) -> np.ndarray:
    """Simple neural network that passes an input through a single hidden layer,
    applies a tanh nonlinearity, and computes the output by applying an affine
    transformation. This representation parameterizes a vector field to use in
    a neural ODE.

    Args:
        z: The input at which to compute the time derivative.
        t: The time at which to compute the time derivative.
        params: The weights and biases at each layer of the network.

    Returns:
        dzdt: The time derivative at the prescribed input.

    """
    inputs = np.hstack((z, t))
    for W, b in params:
        outputs = np.dot(W, inputs) + b
        inputs = np.tanh(outputs)
    dzdt = outputs
    return dzdt

rev_dyn = lambda z, t, params: -node_sample_vector_field(z, 1.0-t, params)
jac_rev_dyn = autograd.jacobian(rev_dyn)

def node_log_prob_vector_field(z_and_div: np.ndarray, t: np.ndarray, params: List[Tuple[np.ndarray]]) -> np.ndarray:
    """The ODE for computing the log-density of the neural ODE requires us to
    augment the sampling vector field with a component corresponding to the
    divergence of the vector field, represented here as the trace of the
    Jacobian of the vector field.

    Args:
        z_and_div: Concatenation of the state and the computed divergence at which
            to compute the time derivative.
        t: The time at which to compute the time derivative.
        params: The weights and biases at each layer of the network.

    Returns:
        dz_and_divdt: The time derivative of the state and the divergence.

    """
    z, div = z_and_div[:-1], z_and_div[-1]
    dzdt = rev_dyn(z, t, params)
    ddivdt = -np.trace(jac_rev_dyn(z, t, params))
    dz_and_divdt = np.hstack((dzdt, ddivdt))
    return dz_and_divdt

def node_sample_solve(z: np.ndarray, params: Tuple[Tuple[np.ndarray]]) -> np.ndarray:
    t = np.array([0.0, 1.0])
    x = odeint(node_sample_vector_field, z, t, tuple((params, )))[-1]
    return x

def node_log_prob_solve(x: np.ndarray, params: List[Tuple[np.ndarray]]) -> Tuple[np.ndarray, float]:
    z_and_div = np.hstack((x, 0.0))
    t = np.array([0.0, 1.0])
    sol = odeint(node_log_prob_vector_field, z_and_div, t, tuple((params, )))[-1]
    z, fldj = sol[:-1], sol[-1]
    return z, fldj

class NODE(Distribution):
    """Implements sampling from the neural ODE normalizing flow architecture. A
    neural ODE is generated by a simple single-layer neural network which
    parameterizes the vector field of the flow.

    """
    def __init__(self, params: List[Tuple[np.ndarray]]):
        self.params = params

    def log_density(self, x: np.ndarray) -> float:
        z, fldj = node_log_prob_solve(x, self.params)
        ld = norm_log_pdf(z, 0.0, 1.0).sum(axis=-1) - fldj
        return ld

    def sample(self) -> np.ndarray:
        k = self.params[0][0].shape[-1] - 1
        z = np.random.normal(size=(k, ))
        x = node_sample_solve(z, self.params)
        return x

node_nll = lambda params, x: -NODE(params).log_density(x)

class MultivariateNormalMixture(Distribution):
    """Mixture of multivariate normal distributions. The mixture of multivariate
    normals is specified by the mean and covariance of each component normal
    distribution as well as the mixture probabilities associated to each
    component.

    Parameters:
        mu: Array of means for each component of the mixture.
        sigma: Array of covariances for each component of the mixture.
        probs: Mixture probabilities.

    """
    def __init__(self, mus: np.ndarray, sigmas: np.ndarray, probs: np.ndarray):
        super().__init__()
        self.comps = [MultivariateNormal(m, s) for m, s in zip(mus, sigmas)]
        self.num_comps = len(self.comps)
        self.probs = probs

    def log_density(self, x: np.ndarray) -> float:
        cld = np.array([c.log_density(x) for c in self.comps])
        ld = spsp.logsumexp(cld + np.log(self.probs))
        return ld

    def sample(self) -> np.ndarray:
        c: MultivariateNormal = np.random.choice(self.comps, p=self.probs)
        x = c.sample()
        return x

def init_params(scale: float, layer_sizes: Tuple[int]) -> List[Tuple[np.ndarray]]:
    """Generates initial parameters for a neural network. The parameters consist of
    a matrix of weights and a vector of biases for each hidden layer of the
    network. Initial parameters are sampled from a normal distribution with the
    prescribed scale.

    Args:
        scale: The standard deviation of the generative Gaussian.
        layer_sizes: The size of the neural network architecture.

    Returns:
        params: Parameterization of a feed-forward neural network.

    """
    params = [
        (np.random.randn(insize, outsize) * scale,   # weight matrix
         np.random.randn(outsize) * scale)           # bias vector
        for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    return params

def nn(inputs: np.ndarray, params: List[Tuple[np.ndarray]]) -> np.ndarray:
    """Given the parameterization of a neural network in terms of its weights and
    biases, applies the neural network to an given array of inputs. The
    non-linearity used in the hidden layers is a rectified linear unit.

    Args:
        inputs: The vectors to provide as input to the neural network.
        params: Parameterization of a feed-forward neural network.

    Returns:
        output: The result of applying the feed-forward neural network to the
            input.

    """
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.maximum(0.0, outputs)
    return outputs

def realnvp_forward(
        z: np.ndarray,
        num_split: int,
        perm: np.ndarray,
        mu_params: List[Tuple[np.ndarray]],
        sigma_params: List[Tuple[np.ndarray]]
) -> np.ndarray:
    """The RealNVP transformation applies an affine transformation to a subset of
    the variables wherein the shift and scale of the affine transformation is
    controlled by the other subset of variables. These shfits and scales are
    parameterized by neural networks. We also apply a permutation of the
    variables in order to modify which variables are given an affine
    transformation and which are held constant.

    Args:
        z: Input to the RealNVP transformation.
        num_split: How many of the variables to hold constant.
        perm: The permutation to apply to the variables.
        mu_params: The neural network parameterization of the affine shift.
        sigma_params: The neural network parameterization of the affine scale.

    Returns:
        o: The output of the RealNVP transformation.

    """
    z = z[..., perm]
    x, y = z[..., :num_split], z[..., num_split:]
    mu = nn(x, mu_params)
    sigma = np.exp(nn(x, sigma_params))
    xp = sigma*y + mu
    o = np.concatenate([x, xp], axis=-1)
    return o

def realnvp_inverse(
        o: np.ndarray,
        num_split: int,
        perm: np.ndarray,
        mu_params: List[Tuple[np.ndarray]],
        sigma_params: List[Tuple[np.ndarray]]
) -> Tuple[np.ndarray]:
    """The inverse of the RealNVP transformation, which undoes the affine
    transformation and permutation of the forward RealNVP procedure. In the
    reverse, we additionally compute the log-determinant of the Jacobian of the
    forward transformation.

    Args:
        o: The output of the RealNVP forward transformation to be inverted.
        num_split: How many of the variables to hold constant.
        perm: The permutation to apply to the variables.
        mu_params: The neural network parameterization of the affine shift.
        sigma_params: The neural network parameterization of the affine scale.

    Returns:
        z: Input to the RealNVP transformation to be recovered.

    """
    x, xp = o[..., :num_split], o[..., num_split:]
    mu = nn(x, mu_params)
    log_sigma = nn(x, sigma_params)
    fldj = np.sum(log_sigma, axis=-1)
    sigma = np.exp(log_sigma)
    y = (xp - mu) / sigma
    z = np.concatenate([x, y], axis=-1)[..., np.argsort(perm)]
    return z, fldj

def realnvp_forward_composition(
        z: np.ndarray,
        mu_params: List[List[Tuple[np.ndarray]]],
        sigma_params: List[List[Tuple[np.ndarray]]],
        perms: List[np.ndarray]
) -> np.ndarray:
    for mup, sigmap, perm in zip(mu_params, sigma_params, perms):
        num_split = mup[0][0].shape[0]
        z = realnvp_forward(z, num_split, perm, mup, sigmap)
    return z

def realnvp_inverse_composition(
        x: np.ndarray,
        mu_params: List[List[Tuple[np.ndarray]]],
        sigma_params: List[List[Tuple[np.ndarray]]],
        perms: List[np.ndarray]
) -> Tuple[np.ndarray]:
    fldj = 0.0
    for mup, sigmap, perm in zip(
            reversed(mu_params),
            reversed(sigma_params),
            reversed(perms)
    ):
        num_split = mup[0][0].shape[0]
        x, d = realnvp_inverse(x, num_split, perm, mup, sigmap)
        fldj += d
    return x, fldj

class RealNVP(Distribution):
    """Implements the RealNVP normalizing flow architecture. The RealNVP consists
    of a series of affine transformations whose shifts and scales are
    parameterized by neural networks. By controlling which variables undergo an
    affine transformation and which are held constant and provided as input to
    the neural networks, one produces a transformation with an analytical
    Jacobian determinant.

    Parameters:
        mu_params: The neural network parameterizations of the affine shift.
        sigma_params: The neural network parameterizations of the affine scale.
        perms: The permutations to apply to the variables.

    """
    def __init__(self,
                 mu_params: List[List[Tuple[np.ndarray]]],
                 sigma_params: List[List[Tuple[np.ndarray]]],
                 perms: List[np.ndarray]
    ):
        self.mu_params = mu_params
        self.sigma_params = sigma_params
        self.perms = perms

    def log_density(self, x: np.ndarray) -> np.ndarray:
        z, fldj = realnvp_inverse_composition(
            x, self.mu_params, self.sigma_params, self.perms)
        ld = norm_log_pdf(z, 0.0, 1.0).sum(axis=-1) - fldj
        return ld

    def sample(self) -> np.ndarray:
        k = len(self.perms[0])
        z = np.random.normal(size=(k, ))
        z = realnvp_forward_composition(
            z, self.mu_params, self.sigma_params, self.perms)
        return z

class NealFunnel(Distribution):
    """Neal's funnel distribution is a hierarchical probability distribution
    wherein one random variable determines the scale of a subsequent random
    variable. The construction of this distribution causes is to exhibit
    multi-scale phenomena, wherein the scale of the variable at the lowest
    level of the hierarchy varies greatly.

    Parameters:
        num_dims: The dimensionality of Neal's funnel distribution at both layers
            of the hierarchy.

    """
    def __init__(self, num_dims: int):
        super().__init__()
        self.num_low = num_dims - 1

    def log_density(self, x: np.ndarray) -> float:
        r, v = x[..., :-1], x[..., -1]
        ld_v = norm_log_pdf(v, 0.0, 9.0)
        ld_r = np.sum(norm_log_pdf(r, 0.0, np.exp(-v)), axis=-1)
        ld = ld_v + ld_r
        return ld

    def sample(self) -> np.ndarray:
        v = 3.0 * np.random.normal()
        r = np.exp(-0.5*v) * np.random.normal(size=(self.num_low, ))
        x = np.hstack((r, v))
        return x


