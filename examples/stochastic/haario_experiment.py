import argparse
import os
import pickle
import time

import arviz
import autograd
import autograd.numpy as np
from autograd.misc import flatten
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

import distributions
import mh
import optimization
import psd
import targets

parser = argparse.ArgumentParser(description='Sampling in a Brownian bridge')
parser.add_argument('--target', type=str, default='multimodal', help='Which target distribution to employ')
parser.add_argument('--num-samples', type=int, default=100000, help='Number of Monte Carlo samples to generate')
args = parser.parse_args()

def loss(params, perms, x, unflatten):
    mu_params, sigma_params = unflatten(params)
    nll = -np.mean(distributions.RealNVP(mu_params, sigma_params, perms).log_density(x))
    return nll

def experiment(target, num_samples):
    if target == 'multimodal':
        target, num_dims = targets.multimodal_target()
    elif target == 'neal-funnel':
        target, num_dims = targets.neal_funnel_target()

    ap = np.zeros(num_samples)
    samples = np.zeros([num_samples, num_dims])
    mu = np.zeros(num_dims)
    sigma = np.eye(num_dims)
    proposal = distributions.MultivariateNormal(mu, sigma)
    curr = mh.State(mu, target.log_density(mu), proposal.log_density(mu))
    num_acc = 0

    sum_x = np.zeros(num_dims)
    sum_xo = np.zeros([num_dims, num_dims])

    start = time.time()
    for i in range(num_samples):
        s = curr.state
        proposal = distributions.MultivariateNormal(s, sigma)
        prop = mh.propose_state(proposal, target)
        curr = mh.State(s, curr.log_target, prop.log_proposal)
        curr, info = mh.metropolis_hastings(prop, curr)
        ap[i] = info.accept_prob
        num_acc += int(info.accepted)
        samples[i] = curr.state

        n = i + 1
        sum_x += curr.state
        sum_xo += np.outer(curr.state, curr.state)
        if num_acc > 100:
            sigma = sum_xo / n - np.outer(sum_x, sum_x) / n**2
            sigma *= 2.38**2 / num_dims

        if (i+1) % 10000 == 0 or i == 0:
            print('iter. {} - acc. prob.: {:.4f}'.format(i+1, num_acc / (i+1)))

    elapsed = time.time() - start
    return samples, ap, elapsed

def main():
    id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items())
    id = id.replace('_', '-')
    samples, ap, elapsed = experiment(
        args.target,
        args.num_samples,
    )

    with open(os.path.join('samples', 'haario-{}.pkl'.format(id)), 'wb') as f:
        pickle.dump({
            'samples': samples,
            'ap': ap,
            'time': elapsed
        }, f)

if __name__ == '__main__':
    main()
