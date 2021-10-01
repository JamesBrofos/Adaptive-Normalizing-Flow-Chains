from abc import abstractmethod, ABC

import numpy as np
import scipy.linalg as spla
import scipy.special as spsp
import scipy.stats as spst


class Distribution(ABC):
    """Implements a basic distribution object. Distribution objects implement
    methods to compute their log-density at a given input and to draw i.i.d.
    samples from them. The log-density may be specified up to an additive
    constant.

    Parameters:
        num_dims: The dimensionality of the distribution.

    """
    def __init__(self, num_dims: int):
        self.num_dims = num_dims

    @abstractmethod
    def log_density(self, x: np.ndarray) -> float:
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> np.ndarray:
        raise NotImplementedError()

def norm_log_pdf(x: np.ndarray, mu: float, sigmasq: float) -> float:
    ld = -0.5*np.square(x - mu) / sigmasq \
        - 0.5*np.log(sigmasq) - 0.5*np.log(2*np.pi)
    return ld

class Normal(Distribution):
    """Normal distribution parameterized by its mean and variance.

    Parameters:
        mu: The mean of the normal distribution.
        sigmasq: The variance of the normal distribution.

    """
    def __init__(self, mu: float, sigmasq: float):
        super().__init__(1)
        self.mu = mu
        self.sigmasq = sigmasq
        self.sigma = np.sqrt(sigmasq)

    def log_density(self, x: np.ndarray) -> float:
        ld = norm_log_pdf(x, self.mu, self.sigmasq)
        return ld

    def sample(self) -> np.ndarray:
        x = self.sigma * np.random.normal() + self.mu
        return x

class MultivariateNormal(Distribution):
    """The multivariate normal distribution is the multivariate generalization of
    the normal distribution. It is parameterized by its mean vector and its
    covariance matrix.

    Parameters:
        mu: The mean vector of the multivariate normal.
        sigma: The covariance matrix of the multivariate normal.

    """
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        super().__init__(len(mu))
        self.mu = mu
        self.sigma = sigma
        self.chol = np.linalg.cholesky(sigma)
        self.log_det = 2*np.sum(np.log(np.diag(self.chol)))
        Id = np.eye(self.num_dims)
        chol_inv = spla.solve_triangular(self.chol, Id, lower=True)
        self.sigma_inv = chol_inv.T@chol_inv

    def log_density(self, x: np.ndarray) -> float:
        b = x - self.mu
        ld = -0.5*b@self.sigma_inv@b - 0.5*self.num_dims*np.log(2*np.pi) \
            -0.5*self.log_det
        return ld

    def sample(self) -> np.ndarray:
        z = np.random.normal(size=self.num_dims)
        x = self.chol@z + self.mu
        return x

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
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, probs: np.ndarray):
        super().__init__(mu.shape[-1])
        self.comps = [MultivariateNormal(m, s) for m, s in zip(mu, sigma)]
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
        super().__init__(num_dims)
        self.num_low = num_dims - 1

    def log_density(self, x: np.ndarray) -> float:
        r, v = x[:-1], x[-1]
        ld_v = norm_log_pdf(v, 0.0, 9.0)
        ld_r = np.sum(norm_log_pdf(r, 0.0, np.exp(-v)))
        ld = ld_v + ld_r
        return ld

    def sample(self) -> np.ndarray:
        v = 3.0 * np.random.normal()
        r = np.exp(-0.5*v) * np.random.normal(size=(self.num_low, ))
        x = np.hstack((r, v))
        return x
