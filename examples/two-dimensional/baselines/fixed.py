import os
from abc import abstractmethod, ABC
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from distributions import MultivariateNormal, MultivariateNormalMixture, NealFunnel


class State(NamedTuple):
    """An object representing a state of the Markov chain, which consists of a
    location in state space, the log-density of the target distribution, and
    the log-density of the proposal distribution.

    """
    state: np.ndarray
    log_target: float
    log_proposal: float

def metropolis_hastings(prop: State, curr: State) -> Tuple[State, bool]:
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
    logu = np.log(np.random.uniform())
    accepted = logu < m
    nxt = prop if accepted else curr
    return nxt, accepted

target = NealFunnel()

def propose_state(proposal: MultivariateNormalMixture) -> State:
    p = proposal.sample()
    prop = State(p, target.log_density(p), proposal.log_density(p))
    return prop

def independent_metropolis_hastings(curr: State, proposal: MultivariateNormalMixture) -> Tuple[State, bool]:
    """Implements the independent Metropolis-Hastings algorithm. Samples a proposal
    state from a given proposal distribution and applies the
    Metropolis-Hastings accept-reject criterion in order to ensure detailed
    balance.

    Args:
        curr: The current state of the Markov chain.
        proposal: The proposal distribution for the independent
            Metropolis-Hastings algorithm.

    Returns:
        nxt: The next state of the Markov chain.
        acc: Whether or not the proposal state was accepted.

    """
    prop = propose_state(proposal)
    nxt, acc = metropolis_hastings(prop, curr)
    return nxt, acc

def main():
    sigma = 10000*np.eye(target.num_dims)
    mu = np.zeros(target.num_dims)
    proposal = MultivariateNormal(mu, sigma)

    curr = State(mu, target.log_density(mu), proposal.log_density(mu))
    num_samples = 1000000
    samples = np.zeros([num_samples, target.num_dims])
    accprob = np.zeros(num_samples)

    for i in tqdm.tqdm(range(num_samples)):
        curr, acc = independent_metropolis_hastings(curr, proposal)
        samples[i] = curr.state
        accprob[i] = acc

    iid = np.array([target.sample() for i in range(num_samples)])
    accprob = np.cumsum(accprob) / (np.arange(num_samples) + 1)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(131)
    ax.plot(iid[:, 0], iid[:, 1], '.')
    ax.grid(linestyle=':')
    ax = fig.add_subplot(132)
    ax.plot(samples[:, 0], samples[:, 1], '.')
    ax.grid(linestyle=':')
    ax = fig.add_subplot(133)
    ax.plot(np.arange(num_samples) + 1, accprob)
    ax.grid(linestyle=':')
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'fixed.png'))


if __name__ == '__main__':
    main()
