import argparse
import os
import pickle
import time

import arviz
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

import distributions
import mh
import psd
import targets

parser = argparse.ArgumentParser(description='Sampling in a Brownian bridge')
parser.add_argument('--num-times', type=int, default=50, help='Number of observations of the Brownian bridge')
parser.add_argument('--step-size', type=float, default=1e-3, help='Step-size for pseudo-likelihood training')
parser.add_argument('--num-samples', type=int, default=1000000, help='Number of Monte Carlo samples to generate')
parser.add_argument('--precondition', dest='precondition', action='store_true', help='Whether or not to precondition the Langevin dynamics')
parser.add_argument('--no-precondition', dest='precondition', action='store_false')
parser.set_defaults(precondition=False)
args = parser.parse_args()


def experiment(num_times, num_samples, step_size, precondition):
    target, num_dims = targets.brownian_bridge_target(num_times)
    rvs = np.array([target.sample() for _ in range(1000)])

    grad_log_density = autograd.grad(target.log_density)
    if precondition:
        drift = lambda x: target.sigma@grad_log_density(x)
        sigma = step_size * target.sigma
    else:
        drift = lambda x: grad_log_density(x)
        sigma = step_size * np.eye(num_dims)

    mu = np.mean(rvs, 0)
    proposal = distributions.MultivariateNormal(
        mu + 0.5*step_size*drift(mu), sigma
    )

    ap = np.zeros(num_samples)
    samples = np.zeros([num_samples, num_times])
    curr = mh.propose_state(proposal, target)
    num_acc = 0

    start = time.time()
    for i in range(num_samples):
        s = curr.state
        proposal = distributions.MultivariateNormal(
            s + 0.5*step_size*drift(s), sigma
        )
        prop = mh.propose_state(proposal, target)
        p = prop.state
        counter = distributions.MultivariateNormal(
            p + 0.5*step_size*drift(p), sigma
        )
        curr = mh.State(s, curr.log_target, counter.log_density(s))
        curr, info = mh.metropolis_hastings(prop, curr)
        ap[i] = info.accept_prob
        num_acc += int(info.accepted)
        samples[i] = curr.state

        if (i+1) % 10000 == 0 or i == 0:
            print('iter. {} - acc. prob.: {:.4f}'.format(i+1, num_acc / (i+1)))

    elapsed = time.time() - start
    return samples, ap, elapsed

def main():
    id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items())
    id = id.replace('_', '-')
    samples, ap, elapsed = experiment(
        args.num_times,
        args.num_samples,
        args.step_size,
        args.precondition
    )

    with open(os.path.join('samples', 'brownian-bridge-langevin-{}.pkl'.format(id)), 'wb') as f:
        pickle.dump({
            'samples': samples,
            'ap': ap,
            'time': elapsed
        }, f)

if __name__ == '__main__':
    main()
