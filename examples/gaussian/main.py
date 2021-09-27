import argparse
import copy
import os
from typing import NamedTuple, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm

parser = argparse.ArgumentParser(description='Violating stationarity with non-independent adaptations')
parser.add_argument('--violate', action='store_true', default=True, help='Violatie stationarity')
parser.add_argument('--no-violate', action='store_false', dest='violate')
parser.add_argument('--severe', action='store_true', default=True, help='Adaptation with severe violation if non-independent')
parser.add_argument('--no-severe', action='store_false', dest='severe')
parser.add_argument('--num-steps', type=int, default=5, help='Number of transition kernel steps')
parser.add_argument('--num-trials', type=int, default=100000, help='Number of Monte Carlo trials')
args = parser.parse_args()


class Norm:
    """Class representing a normal distribution, which includes convenience
    functions for working with the normal distribution. This includes computing
    the log-density of the distribution, generating random variables, and
    computing the KL divergence.

    Parameters:
        mean: The mean of the normal distribution.
        scale: The standard deviation (scale) of the normal distribution.

    """
    def __init__(self, mean: np.ndarray, log_scale: np.ndarray):
        self.mean = mean
        self.log_scale = log_scale

    @property
    def scale(self):
        scale = np.exp(self.log_scale)
        return scale

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        lp = -0.5*np.square(z) - 0.5*np.log(2*np.pi) - np.log(self.scale)
        return lp

    def pdf(self, x: np.ndarray) -> np.ndarray:
        p = np.exp(self.logpdf(x))
        return p

    def rvs(self, size: Optional[Sequence[int]]=None):
        z = np.random.normal(size=size)
        x = self.mean + self.scale*z
        return x

    def kl(self, other) -> Tuple[float, float, float]:
        ma, sa = self.mean, self.scale
        mb, sb = other.mean, other.scale
        va, vb = sa**2, sb**2
        log_sa, log_sb = self.log_scale, other.log_scale
        delta = ma - mb
        kl = np.square(delta) / (2*vb) + 0.5*(va / vb - 1) - log_sa + log_sb
        grad_mean = delta / vb
        grad_log_scale = va / vb - 1
        return kl

class State(NamedTuple):
    """An object representing a state of the Markov chain, which consists of a
    location in state space, the log-density of the target distribution, and
    the log-density of the proposal distribution.

    Parameters:
        state: The location in state space.
        log_target: The log-density of the target distribution.
        log_proposal: The log-density of the proposal distribution.

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

def propose(proposal: Norm, target: Norm) -> State:
    """Generates a proposal given a proposal distribution and a target
    distribution. Computes the log-density of the proposal density and the
    target density at the generated location in state space.

    Args:
        proposal: Proposal distribution from which to generate a proposal state.
        target: The target distributon from which to sample.

    Returns:
        state: The proposal state.

    """
    prop = proposal.rvs()
    log_target = target.logpdf(prop)
    log_proposal = proposal.logpdf(prop)
    state = State(prop, log_target, log_proposal)
    return state

class ChainInfo(NamedTuple):
    """Object to keep track of diagnostic information computed during the
    chain transition.

    Parameters:
        accepted: Whether or not the proposal was accepted.
        kl_tp: The KL divergence between the target distribution and the
            proposal distribution.
        kl_pt: The KL divergence between the proposal distribution and the
            target distribution.

    """
    accepted: bool
    kl_tp: float
    kl_pt: float

class IndependentMetropolisHastings:
    """Implements the independent Metropolis-Hastings algorithm. Samples a proposal
    state from a given proposal distribution and applies the
    Metropolis-Hastings accept-reject criterion in order to ensure detailed
    balance.

    Parameters:
        target: The target distribution from which to sample.

    """
    def __init__(self, target: Norm):
        self.target = target

    def __call__(self, curr: State, proposal: Norm) -> Tuple[State, ChainInfo]:
        prop = propose(proposal, self.target)
        nxt, acc = metropolis_hastings(prop, curr)
        kl_tp = self.target.kl(proposal)
        kl_pt = proposal.kl(self.target)
        info = ChainInfo(acc, kl_tp, kl_pt)
        return nxt, info


def adaptive_stationarity():
    target = Norm(1.0, np.log(0.5))
    proposal = Norm(2.0, np.log(5.0))
    mh = IndependentMetropolisHastings(target)
    chain = np.zeros([args.num_trials])

    for i in tqdm.tqdm(range(args.num_trials)):
        state = target.rvs()
        proposal = Norm(2.0, np.log(5.0))
        curr = State(state, target.logpdf(state), proposal.logpdf(state))
        hist = []
        for j in range(args.num_steps):
            nxt, info = mh(curr, proposal)
            if len(hist) >= 1:
                if args.severe:
                    proposal.mean = np.exp(hist[-1])
                    proposal.log_scale = -2*hist[-1]
                else:
                    proposal.mean = np.mean(hist)
                    std = np.std(hist)
                    if std > 1e-6:
                        proposal.log_scale = np.log(std)
                state = nxt.state
                nxt = State(state, nxt.log_target, proposal.logpdf(state))
            if info.accepted or args.violate:
                hist.append(copy.deepcopy(curr.state))
            curr = copy.deepcopy(nxt)
        chain[i] = curr.state

    r = np.linspace(-3.0, 10.0, 1000)
    plt.figure()
    plt.hist(chain, bins=50, density=True)
    plt.plot(r, target.pdf(r), '--', label='Target')
    plt.grid(linestyle=':')
    plt.ylim((0.0, 0.9))
    plt.savefig(os.path.join('images', 'adaptive-stationarity-violate-{}-severe-{}-num-steps-{}-num-trials-{}.png'.format(args.violate, args.severe, args.num_steps, args.num_trials)))


if __name__ == '__main__':
    adaptive_stationarity()
