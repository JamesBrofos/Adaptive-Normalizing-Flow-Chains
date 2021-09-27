from typing import NamedTuple
import jax.numpy as jnp


class State(NamedTuple):
    """Object to store the state of the Markov chain. This class is used to
    represent both the proposal and the current state of the chain.

    Parameters:
        state: State of the Markov chain.
        log_appox: The log-density of the proposal under the approximate
            distribution.
        log_target: The log-density of the target distribution evaluated at the
            proposal location.

    """
    state: jnp.ndarray
    log_approx: float
    log_target: float
