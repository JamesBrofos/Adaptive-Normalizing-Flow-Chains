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
parser.add_argument('--step-size', type=float, default=1e-5, help='Step-size for pseudo-likelihood training')
parser.add_argument('--step-size-decay', type=float, default=0.99999, help='Decay rate for step-size')
parser.add_argument('--num-samples', type=int, default=1000000, help='Number of Monte Carlo samples to generate')
parser.add_argument('--exact', dest='exact', action='store_true', help='Whether or not to employ the exact forward KL divergence in learning')
parser.add_argument('--no-exact', dest='exact', action='store_false')
parser.set_defaults(exact=False)
args = parser.parse_args()


def experiment(num_times, num_samples, step_size, step_size_decay, exact_kl):
    target, num_dims = targets.brownian_bridge_target(num_times)
    rvs = np.array([target.sample() for _ in range(1000)])

    mu = np.mean(rvs, 0)
    sigma = np.cov(rvs.T)
    proposal = distributions.MultivariateNormal(mu, sigma)

    klpq_grad = autograd.grad(distributions.mvn_kl, (2, 3))
    nll_grad = autograd.grad(distributions.mvn_nll, (0, 1))

    klqp = np.zeros(num_samples)
    klpq = np.zeros(num_samples)
    ap = np.zeros(num_samples)
    num_acc = 0

    samples = np.zeros([num_samples, num_dims])
    curr = mh.propose_state(proposal, target)

    start = time.time()
    for i in range(num_samples):
        step_size *= step_size_decay
        if exact_kl:
            grad_mu, grad_sigma = klpq_grad(target.mu, target.sigma, mu, sigma)
        else:
            x = samples[np.random.randint(i+1, size=100)]
            grad_mu, grad_sigma = nll_grad(mu, sigma, x)
        sigma_inv = proposal.sigma_inv
        rgrad = psd.psd_egrad_to_rgrad(sigma, grad_sigma)
        sigma = psd.psd_retraction(sigma, sigma_inv, -step_size*rgrad)
        mu -= step_size*grad_mu

        proposal = distributions.MultivariateNormal(mu, sigma)
        s = curr.state
        curr = mh.State(s, curr.log_target, proposal.log_density(s))
        prop = mh.propose_state(proposal, target)
        curr, info = mh.metropolis_hastings(prop, curr)

        klqp[i] = proposal.kl(target)
        klpq[i] = target.kl(proposal)
        ap[i] = info.accept_prob
        samples[i] = curr.state
        num_acc += int(info.accepted)

        if (i+1) % 1000 == 0 or i == 0:
            print('iter. {} - klpq: {:.4f} - acc. prob.: {:.4f}'.format(i+1, klpq[i], num_acc / (i+1)))

    elapsed = time.time() - start
    return samples, klpq, ap, elapsed

def main():
    id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items())
    id = id.replace('_', '-')
    samples, klpq, ap, elapsed = experiment(
        args.num_times,
        args.num_samples,
        args.step_size,
        args.step_size_decay,
        args.exact
    )

    with open(os.path.join('samples', 'brownian-bridge-adaptive-{}.pkl'.format(id)), 'wb') as f:
        pickle.dump({
            'samples': samples,
            'klpq': klpq,
            'ap': ap,
            'time': elapsed
        }, f)

if __name__ == '__main__':
    main()


# plt.figure()
# plt.semilogy(np.arange(num_samples) + 1, np.abs(klpq), label='Forward')
# plt.semilogy(np.arange(num_samples) + 1, np.abs(klqp), label='Reverse')
# plt.grid(linestyle=':')
# plt.legend()
# plt.xlabel('Sampling Iteration')
# plt.ylabel('KL Divergence')
# plt.savefig(os.path.join('images', 'brownian-bridge-independent-kl.png'))

# w = 100
# r = np.arange(num_samples) + 1
# roll = pd.Series(ap).rolling(window=w).mean()
# plt.figure()
# plt.plot(r, np.cumsum(ap) / r)
# plt.plot(r, roll)
# plt.grid(linestyle=':')
# plt.xlabel('Sampling Iteration')
# plt.ylabel('Acceptance Probability')
# plt.savefig(os.path.join('images', 'brownian-bridge-independent-ap.png'))

# burn = int(0.1*num_samples)
# ess = np.array([arviz.ess(samples[burn:, i]) for i in range(num_times)])

# plt.figure()
# plt.boxplot(ess, vert=False)
# plt.grid(linestyle=':')
# plt.xlabel('Effective Sample Size')
# plt.savefig(os.path.join('images', 'brownian-bridge-independent-ess.png'))

