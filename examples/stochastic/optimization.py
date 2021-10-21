import autograd.numpy as np


def rmsprop(grad, x, avg_sq_grad, step_size, gamma=0.9, eps=10**-8):
    """Root mean squared prop. See Adagrad paper for details.

    Args:
        grad: The gradients of the loss function to use to update the parameters.
        x: The current value of the parameters.
        avg_sq_grad: A moving average of the squared gradients, used to regularize
            the update to the parameters.
        gamma: Moving average weight.
        eps: Fudge factor to prevent division by zero when regularizing the
            gradient update.

    Returns:
        x: The updated parameters
        avg_sq_grad: Updated moving average of the squared gradients.

    """
    avg_sq_grad = avg_sq_grad * gamma + grad**2 * (1 - gamma)
    x = x - step_size * grad / (np.sqrt(avg_sq_grad) + eps)
    return x, avg_sq_grad

