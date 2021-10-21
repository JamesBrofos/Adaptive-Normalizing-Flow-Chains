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

parser = argparse.ArgumentParser(description='Sampling in simple target distributions')
parser.add_argument('--target', type=str, default='multimodal', help='Which target distribution to employ')
parser.add_argument('--step-size', type=float, default=1e-1, help='Step-size for pseudo-likelihood training')
parser.add_argument('--num-samples', type=int, default=100000, help='Number of Monte Carlo samples to generate')
args = parser.parse_args()


def experiment(target, num_samples, step_size):
    if target == 'multimodal':
        target, num_dims = targets.multimodal_target()
    elif target == 'neal-funnel':
        target, num_dims = targets.neal_funnel_target()
    rvs = np.array([target.sample() for _ in range(1000)])

    grad_log_density = autograd.grad(target.log_density)
    drift = lambda x: grad_log_density(x)
    sigma = step_size * np.eye(num_dims)

    mu = np.mean(rvs, 0)
    proposal = distributions.MultivariateNormal(
        mu + 0.5*step_size*drift(mu), sigma
    )

    ap = np.zeros(num_samples)
    samples = np.zeros([num_samples, num_dims])
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
        args.target,
        args.num_samples,
        args.step_size,
    )

    with open(os.path.join('samples', 'langevin-{}.pkl'.format(id)), 'wb') as f:
        pickle.dump({
            'samples': samples,
            'ap': ap,
            'time': elapsed
        }, f)

if __name__ == '__main__':
    main()
