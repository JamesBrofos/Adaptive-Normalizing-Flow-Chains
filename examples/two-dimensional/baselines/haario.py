import os
from abc import abstractmethod, ABC
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from distributions import MultivariateNormal, MultivariateNormalMixture, NealFunnel


class State(NamedTuple):
    """An object representing a state of the Markov chain, which consists of a
    location in state space and the log-density of the target distribution.

    """
    state: np.ndarray
    log_target: float

def metropolis_hastings(prop: State, curr: State) -> Tuple[State, bool]:
    """Computes and applies the Metropolis accept-reject criterion to ensure that a
    Markov chain satisfies the detailed balance condition.

    Args:
        prop: The proposal state.
        curr: The current state of the Markov chain.

    Returns:
        nxt: The next state of the Markov chain.
        accepted: Whether or not the proposal was accepted.

    """
    m = prop.log_target - curr.log_target
    logu = np.log(np.random.uniform())
    accepted = logu < m
    nxt = prop if accepted else curr
    return nxt, accepted

target = NealFunnel(11)

def propose_state(proposal: MultivariateNormal) -> State:
    p = proposal.sample()
    prop = State(p, target.log_density(p))
    return prop

def gauss_metropolis(curr: State, proposal: MultivariateNormal) -> Tuple[State, bool]:
    """The Gauss-Metropolis Markov chain Monte Carlo procedure uses a multivariate
    normal proposal distribution in order to propose candidate states of the
    chain. This produces a symmetric proposal so that the Metropolis
    accept-reject criterion consists only of the ratio of the target
    distribution at the current and proposal states.

    Args:
        curr: The current state of the Markov chain.
        prop: The proposal state.

    Returns:
        nxt: The next state of the Markov chain.
        acc: Whether or not the proposal state was accepted.

    """
    prop = propose_state(proposal)
    nxt, acc = metropolis_hastings(prop, curr)
    return nxt, acc

def main():
    sigma = np.eye(target.num_dims)
    mu = np.zeros(target.num_dims)
    proposal = MultivariateNormal(mu, sigma)

    curr = State(mu, target.log_density(mu))
    num_samples = 1000000
    samples = np.zeros([num_samples, target.num_dims])
    accprob = np.zeros(num_samples)

    for i in tqdm.tqdm(range(num_samples)):
        curr, acc = gauss_metropolis(curr, proposal)
        samples[i] = curr.state
        accprob[i] = acc
        if i > 0:
            w = 1 / (i + 1)
            sigma = w*np.outer(curr.state, curr.state) + (1-w)*sigma
        proposal = MultivariateNormal(curr.state, sigma)

    iid = np.array([target.sample() for i in range(num_samples)])
    accprob = np.cumsum(accprob) / (np.arange(num_samples) + 1)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(131)
    ax.plot(iid[:, 0], iid[:, -1], '.')
    ax.grid(linestyle=':')
    ax = fig.add_subplot(132)
    ax.plot(samples[:, 0], samples[:, -1], '.')
    ax.grid(linestyle=':')
    ax = fig.add_subplot(133)
    ax.plot(np.arange(num_samples) + 1, accprob)
    ax.grid(linestyle=':')
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'haario.png'))

if __name__ == '__main__':
    main()
