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
parser.add_argument('--num-samples', type=int, default=1000000, help='Number of Monte Carlo samples to generate')
args = parser.parse_args()


def experiment(num_times, num_samples):
    target, num_dims = targets.brownian_bridge_target(num_times)
    rvs = np.array([target.sample() for _ in range(1000)])

    mu = np.mean(rvs, 0)
    sigma = 2.38**2 * np.cov(rvs.T) / num_times
    proposal = distributions.MultivariateNormal(mu, sigma)

    coverr = np.zeros(num_samples)*np.nan
    ap = np.zeros(num_samples)
    samples = np.zeros([num_samples, num_times])
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
        if num_acc > num_times:
            sigma = sum_xo / n - np.outer(sum_x, sum_x) / n**2
            coverr[i] = np.linalg.norm(target.sigma - sigma)
            sigma *= 2.38**2 / num_times

        if (i+1) % 10000 == 0 or i == 0:
            print('iter. {} - acc. prob.: {:.4f}'.format(i+1, num_acc / (i+1)))

    elapsed = time.time() - start
    return samples, coverr, ap, elapsed

def main():
    id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items())
    id = id.replace('_', '-')
    samples, coverr, ap, elapsed = experiment(
        args.num_times,
        args.num_samples
    )

    with open(os.path.join('samples', 'brownian-bridge-haario-{}.pkl'.format(id)), 'wb') as f:
        pickle.dump({
            'samples': samples,
            'coverr': coverr,
            'ap': ap,
            'time': elapsed
        }, f)

if __name__ == '__main__':
    main()

# t = np.linspace(0.0, 1.0, 52)[1:-1]
# num_times = len(t)
# mean = np.sin(np.pi*t)
# cov = distributions.brownian_bridge_covariance(t)
# target = distributions.MultivariateNormal(mean, cov)
# target = distributions.MultivariateNormal(mean, cov)
# rvs = np.array([target.sample() for _ in range(1000)])

# mu = np.mean(rvs, 0)
# sigma = 2.38**2 * np.cov(rvs.T) / num_times
# proposal = distributions.MultivariateNormal(mu, sigma)

# num_samples = 1000000
# coverr = np.zeros(num_samples)*np.nan
# ap = np.zeros(num_samples)
# samples = np.zeros([num_samples, num_times])
# curr = mh.State(mu, target.log_density(mu), proposal.log_density(mu))
# num_acc = 0

# sum_x = 0
# sum_xo = 0

# with tqdm.tqdm(total=num_samples) as pbar:
#     for i in range(num_samples):
#         s = curr.state
#         proposal = distributions.MultivariateNormal(s, sigma)
#         prop = mh.propose_state(proposal, target)
#         curr = mh.State(s, curr.log_target, prop.log_proposal)
#         curr, info = mh.metropolis_hastings(prop, curr)
#         ap[i] = info.accept_prob
#         num_acc += int(info.accepted)
#         samples[i] = curr.state

#         n = i + 1
#         sum_x += curr.state
#         sum_xo += np.outer(curr.state, curr.state)
#         if num_acc > num_times:
#             sigma = sum_xo / n - np.outer(sum_x, sum_x) / n**2
#             coverr[i] = np.linalg.norm(cov - sigma)
#             sigma *= 2.38**2 / num_times

#         pbar.set_postfix({'acc. prob.': num_acc / (i + 1)})
#         pbar.update(1)

# plt.figure()
# plt.semilogy(np.arange(num_samples) + 1, coverr)
# plt.grid(linestyle=':')
# plt.xlabel('Sampling Iteration')
# plt.ylabel('Covariance Error')
# plt.tight_layout()
# plt.savefig(os.path.join('images', 'brownian-bridge-haario-cov.png'))

# w = 100
# r = np.arange(num_samples) + 1
# roll = pd.Series(ap).rolling(window=w).mean()
# plt.figure()
# plt.plot(r, np.cumsum(ap) / r)
# plt.plot(r, roll)
# plt.grid(linestyle=':')
# plt.xlabel('Sampling Iteration')
# plt.ylabel('Acceptance Probability')
# plt.savefig(os.path.join('images', 'brownian-bridge-haario-ap.png'))

# burn = int(0.1*num_samples)
# ess = np.array([arviz.ess(samples[burn:, i]) for i in range(num_times)])

# plt.figure()
# plt.boxplot(ess, vert=False)
# plt.grid(linestyle=':')
# plt.xlabel('Effective Sample Size')
# plt.savefig(os.path.join('images', 'brownian-bridge-haario-ess.png'))

