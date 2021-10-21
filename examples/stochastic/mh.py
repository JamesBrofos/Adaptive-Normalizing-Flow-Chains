from typing import NamedTuple, Tuple

import autograd.numpy as np

from distributions import Distribution


class State(NamedTuple):
    """An object representing a state of the Markov chain, which consists of a
    location in state space, the log-density of the target distribution, and
    the log-density of the proposal distribution.

    """
    state: np.ndarray
    log_target: float
    log_proposal: float

class MetropolisHastingsInfo(NamedTuple):
    """An object storing diagnostic information about the Metropolis-Hastings
    accept-reject decision.

    """
    accept_prob: float
    accepted: bool

def metropolis_hastings(prop: State, curr: State) -> Tuple[State, MetropolisHastingsInfo]:
    """Computes and applies the Metropolis-Hastings accept-reject criterion to
    ensure that a Markov chain satisfies the detailed balance condition.

    Args:
        prop: The proposal state.
        curr: The current state of the Markov chain.

    Returns:
        nxt: The next state of the Markov chain.
        accepted: Whether or not the proposal was accepted.

    """
    m = prop.log_target - curr.log_target + curr.log_proposal - prop.log_proposal
    m = np.minimum(m, 0.0)
    log_u = np.log(np.random.uniform())
    accepted = log_u < m
    nxt = prop if accepted else curr
    info = MetropolisHastingsInfo(np.exp(m), accepted)
    return nxt, info

def propose_state(proposal: Distribution, target: Distribution) -> State:
    """Given a proposal distribution, computes a proposal state by sampling the
    proposal distribution and recording the log-density of the target
    distribution at the candidate location and the log-density of the proposal
    distribution.

    Args:
        proposal: The proposal distribution used in the Metropolis-Hastings
            decision.
        target: The target distribution from which to draw samples.

    Returns:
        prop: The proposal state of the Markov chain, which will be accepted or
            rejected according to the Metropolis-Hastings criterion.

    """
    p = proposal.sample()
    prop = State(p, target.log_density(p), proposal.log_density(p))
    return prop
