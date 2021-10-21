import autograd.numpy as np


symm = lambda x: 0.5*(x + np.swapaxes(x, -1, -2))

def psd_egrad_to_rgrad(sigma: np.ndarray, egrad: np.ndarray) -> np.ndarray:
    """Converts the Euclidean gradient with respect to a positive definite matrix
    into a vector in the tangent space of the positive definite matrix
    manifold.

    Args:
        sigma: Positive definite matrix on the manifold.
        egrad: Euclidean gradient with respect to sigma.

    Returns:
        rgrad: The Riemannian gradient in the tangent space of the positive
            definite matrix manifold.

    """
    rgrad = sigma@symm(egrad)@sigma
    return rgrad

def psd_retraction(sigma: np.ndarray, sigma_inv: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Applies a second-order retraction to the positive definite matrix `sigma` to
    move in the direction `tau`.

    Args:
        sigma: Positive definite matrix on the manifold.
        sigma_inv: The matrix inverse of the positive definite matrix.
        tau: Direction in the tangent space of the manifold in which to move.

    Returns:
        new_sigma: An updated positive definite matrix.

    """
    eta = sigma_inv@tau
    new_sigma = symm(sigma + eta + 0.5*eta@sigma_inv@eta)
    return new_sigma
