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

parser = argparse.ArgumentParser(description='Sampling in simple target distributions')
parser.add_argument('--target', type=str, default='multimodal', help='Which target distribution to employ')
parser.add_argument('--step-size', type=float, default=1e-3, help='Step-size for pseudo-likelihood training')
parser.add_argument('--step-size-decay', type=float, default=0.9999, help='Decay rate for step-size')
parser.add_argument('--num-samples', type=int, default=100000, help='Number of Monte Carlo samples to generate')
args = parser.parse_args()

def loss(params, perms, x, unflatten):
    mu_params, sigma_params = unflatten(params)
    nll = -np.mean(distributions.RealNVP(mu_params, sigma_params, perms).log_density(x))
    return nll

def experiment(target, num_samples, step_size, step_size_decay):
    if target == 'multimodal':
        m = 8
        target, num_dims = targets.multimodal_target()
    elif target == 'neal-funnel':
        m = 3
        target, num_dims = targets.neal_funnel_target()

    perms = [np.array([1, 0]) for _ in range(m)]
    mu_params = [distributions.init_params(0.01, [1, 64, 1]) for _ in range(m)]
    sigma_params = [distributions.init_params(0.01, [1, 64, 1]) for _ in range(m)]
    params, unflatten = flatten([mu_params, sigma_params])


    samples = np.zeros([num_samples, 2])
    ap = np.zeros(num_samples)
    losses = np.zeros(num_samples)
    proposal = distributions.RealNVP(mu_params, sigma_params, perms)
    curr = mh.propose_state(proposal, target)
    num_acc = 0
    avg_sq_grad = np.ones(len(params))

    start = time.time()
    for i in range(num_samples):
        if True:
            step_size *= step_size_decay
            x = samples[np.random.randint(i+1, size=100)]
            value, grad = autograd.value_and_grad(loss)(params, perms, x, unflatten)
            grad = np.clip(grad, -1.0, 1.0)
            grad = np.where(np.isnan(grad), 0.0, grad)
            params, avg_sq_grad = optimization.rmsprop(grad, params, avg_sq_grad, step_size)
            mu_params, sigma_params = unflatten(params)
            proposal = distributions.RealNVP(mu_params, sigma_params, perms)
        s = curr.state
        curr = mh.State(s, curr.log_target, proposal.log_density(s))
        prop = mh.propose_state(proposal, target)
        curr, info = mh.metropolis_hastings(prop, curr)
        ap[i] = info.accept_prob
        samples[i] = curr.state
        losses[i] = value
        num_acc += int(info.accepted)

        if (i+1) % 100 == 0:
            print('iter. {} - loss: {:.4f} - acc. prob.: {:.4f}'.format(i+1, value, num_acc / (i+1)))

            plt.figure()
            plt.plot(x[:, 0], x[:, 1], '.')
            plt.grid(linestyle=':')
            plt.tight_layout()
            plt.savefig(os.path.join('images', 'adaptive.png'))
            plt.close()

    elapsed = time.time() - start
    return samples, losses, ap, elapsed

def main():
    id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items())
    id = id.replace('_', '-')
    samples, losses, ap, elapsed = experiment(
        args.target,
        args.num_samples,
        args.step_size,
        args.step_size_decay,
    )

    with open(os.path.join('samples', 'adaptive-{}.pkl'.format(id)), 'wb') as f:
        pickle.dump({
            'samples': samples,
            'ap': ap,
            'losses': losses,
            'time': elapsed
        }, f)

if __name__ == '__main__':
    main()
