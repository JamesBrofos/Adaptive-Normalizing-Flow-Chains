import os
import pickle

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as spst

import targets

with open(os.path.join('samples', 'brownian-bridge-haario-num-times-50-num-samples-1000000.pkl'), 'rb') as f:
    h = pickle.load(f)

with open(os.path.join('samples', 'brownian-bridge-langevin-num-times-50-step-size-0.001-num-samples-1000000-precondition-False.pkl'), 'rb') as f:
    l = pickle.load(f)

with open(os.path.join('samples', 'brownian-bridge-langevin-num-times-50-step-size-0.2-num-samples-1000000-precondition-True.pkl'), 'rb') as f:
    p = pickle.load(f)

with open(os.path.join('samples', 'brownian-bridge-adaptive-num-times-50-step-size-1e-05-step-size-decay-0.99999-num-samples-1000000-exact-True.pkl'), 'rb') as f:
    e = pickle.load(f)

with open(os.path.join('samples', 'brownian-bridge-adaptive-num-times-50-step-size-1e-05-step-size-decay-0.99999-num-samples-1000000-exact-False.pkl'), 'rb') as f:
    a = pickle.load(f)


num_times = 50
burn = 100000
ess = [
    np.array([arviz.ess(h['samples'][burn:, i]) for i in range(num_times)]),
    np.array([arviz.ess(l['samples'][burn:, i]) for i in range(num_times)]),
    np.array([arviz.ess(p['samples'][burn:, i]) for i in range(num_times)]),
    np.array([arviz.ess(e['samples'][burn:, i]) for i in range(num_times)]),
    np.array([arviz.ess(a['samples'][burn:, i]) for i in range(num_times)])
]
ess_per_sec = [
    ess[0] / h['time'],
    ess[1] / l['time'],
    ess[2] / p['time'],
    ess[3] / e['time'],
    ess[4] / a['time']
]

plt.figure()
plt.boxplot(ess, vert=False)
plt.grid(linestyle=':')
plt.yticks([1, 2, 3, 4, 5], ['Haario\n(R.W.M.)', 'Langevin', 'Langevin\n(Precond.)', 'KL Flow\n(I.M.H.)', 'Pseudo-Likelihood\n(I.M.H.)'], fontsize=20)
plt.xlabel('Effective Sample Size', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join('images', 'brownian-bridge-ess.png'))

plt.figure()
plt.boxplot(ess_per_sec, vert=False)
plt.grid(linestyle=':')
plt.yticks([1, 2, 3, 4, 5], ['Haario\n(R.W.M.)', 'Langevin', 'Langevin\n(Precond.)', 'KL Flow\n(I.M.H.)', 'Pseudo-Likelihood\n(I.M.H.)'], fontsize=20)
plt.xlabel('Effective Sample Size per Second', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join('images', 'brownian-bridge-ess-per-second.png'))

target = targets.brownian_bridge_target(50)[0]
iid = np.array([target.sample() for _ in range(1000000)])
ks = []
for m in (h, l, p, e, a):
    stats = np.zeros(100)
    for i in range(len(stats)):
        u = np.random.normal(size=(50, ))
        u = u / np.linalg.norm(u)
        stats[i] = spst.ks_2samp(m['samples']@u, iid@u).statistic
    ks.append(stats)

plt.figure()
plt.boxplot(ks, vert=False)
plt.grid(linestyle=':')
plt.yticks([1, 2, 3, 4, 5], ['Haario\n(R.W.M.)', 'Langevin', 'Langevin\n(Precond.)', 'KL Flow\n(I.M.H.)', 'Pseudo-Likelihood\n(I.M.H.)'], fontsize=20)
plt.xlabel('Kolmogorov-Smirnov Statistic', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join('images', 'brownian-bridge-ks.png'))


num_samples = 1000000
w = 1000
r = np.arange(num_samples) + 1
plt.figure()
plt.plot(r, pd.Series(h['ap']).rolling(window=w).mean(), label='Haario')
plt.plot(r, pd.Series(a['ap']).rolling(window=w).mean(), label='Pseudo-Likelihood')
plt.plot(r, pd.Series(e['ap']).rolling(window=w).mean(), label='KL Flow')
plt.legend(fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Sampling Iteration', fontsize=20)
plt.ylabel('Acceptance Probability', fontsize=20)
plt.savefig(os.path.join('images', 'brownian-bridge-ap.png'))

