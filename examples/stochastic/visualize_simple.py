import os
import pickle

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as spst

import targets

with open(os.path.join('samples', 'haario-target-multimodal-num-samples-100000.pkl'), 'rb') as f:
    h = pickle.load(f)

with open(os.path.join('samples', 'adaptive-target-multimodal-step-size-0.001-step-size-decay-0.9999-num-samples-100000.pkl'), 'rb') as f:
    a = pickle.load(f)

with open(os.path.join('samples', 'langevin-target-multimodal-step-size-0.1-num-samples-100000.pkl'), 'rb') as f:
    l = pickle.load(f)

burn = 1000
ess = [
    np.array([arviz.ess(h['samples'][burn:, i]) for i in range(2)]),
    np.array([arviz.ess(a['samples'][burn:, i]) for i in range(2)]),
    np.array([arviz.ess(l['samples'][burn:, i]) for i in range(2)]),
]
ess_per_sec = [
    np.array([arviz.ess(h['samples'][burn:, i]) for i in range(2)]) / h['time'],
    np.array([arviz.ess(a['samples'][burn:, i]) for i in range(2)]) / a['time'],
    np.array([arviz.ess(l['samples'][burn:, i]) for i in range(2)]) / l['time'],
]

plt.figure()
plt.boxplot(ess_per_sec, vert=False)
plt.grid(linestyle=':')
plt.yticks([1, 2, 3], ['Haario\n(R.W.M.)', 'Pseudo-Likelihood\n(I.M.H.)', 'Langevin'], fontsize=20)
plt.xlabel('Effective Sample Size per Second', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join('images', 'multimodal-ess-per-sec.png'))

target = targets.multimodal_target()[0]
iid = np.array([target.sample() for _ in range(100000)])
ks = []
for m in (h, a, l):
    stats = np.zeros(100)
    for i in range(len(stats)):
        u = np.random.normal(size=(2, ))
        u = u / np.linalg.norm(u)
        stats[i] = spst.ks_2samp(m['samples']@u, iid@u).statistic
    ks.append(stats)

plt.figure()
plt.boxplot(ks, vert=False)
plt.grid(linestyle=':')
plt.yticks([1, 2, 3], ['Haario\n(R.W.M.)', 'Pseudo-Likelihood\n(I.M.H.)', 'Langevin'], fontsize=20)
plt.xlabel('Kolmogorov-Smirnov Statistic', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join('images', 'multimodal-ks.png'))

num_samples = 100000
w = 1000
r = np.arange(num_samples) + 1
plt.figure()
plt.plot(r, pd.Series(h['ap']).rolling(window=w).mean(), label='Haario')
plt.plot(r, pd.Series(a['ap']).rolling(window=w).mean(), label='Pseudo-Likelihood')
plt.legend(fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Sampling Iteration', fontsize=20)
plt.ylabel('Acceptance Probability', fontsize=20)
plt.savefig(os.path.join('images', 'multimodal-ap.png'))


with open(os.path.join('samples', 'haario-target-neal-funnel-num-samples-100000.pkl'), 'rb') as f:
    h = pickle.load(f)

with open(os.path.join('samples', 'adaptive-target-neal-funnel-step-size-0.001-step-size-decay-0.9999-num-samples-100000.pkl'), 'rb') as f:
    a = pickle.load(f)

with open(os.path.join('samples', 'langevin-target-neal-funnel-step-size-0.1-num-samples-100000.pkl'), 'rb') as f:
    l = pickle.load(f)


target = targets.neal_funnel_target()[0]
iid = np.array([target.sample() for _ in range(100000)])
ks = []
for m in (h, a, l):
    stats = np.zeros(100)
    for i in range(len(stats)):
        u = np.random.normal(size=(2, ))
        u = u / np.linalg.norm(u)
        stats[i] = spst.ks_2samp(m['samples']@u, iid@u).statistic
    ks.append(stats)

plt.figure()
plt.boxplot(ks, vert=False)
plt.grid(linestyle=':')
plt.yticks([1, 2, 3], ['Haario\n(R.W.M.)', 'Pseudo-Likelihood\n(I.M.H.)', 'Langevin'], fontsize=20)
plt.xlabel('Kolmogorov-Smirnov Statistic', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join('images', 'neal-funnel-ks.png'))

num_samples = 100000
w = 1000
r = np.arange(num_samples) + 1
plt.figure()
plt.plot(r, pd.Series(h['ap']).rolling(window=w).mean(), label='Haario')
plt.plot(r, pd.Series(a['ap']).rolling(window=w).mean(), label='Pseudo-Likelihood')
plt.legend(fontsize=20)
plt.grid(linestyle=':')
plt.xlabel('Sampling Iteration', fontsize=20)
plt.ylabel('Acceptance Probability', fontsize=20)
plt.savefig(os.path.join('images', 'neal-funnel-ap.png'))

burn = 1000
ess = [
    np.array([arviz.ess(h['samples'][burn:, i]) for i in range(2)]),
    np.array([arviz.ess(a['samples'][burn:, i]) for i in range(2)]),
    np.array([arviz.ess(l['samples'][burn:, i]) for i in range(2)]),
]
ess_per_sec = [
    np.array([arviz.ess(h['samples'][burn:, i]) for i in range(2)]) / h['time'],
    np.array([arviz.ess(a['samples'][burn:, i]) for i in range(2)]) / a['time'],
    np.array([arviz.ess(l['samples'][burn:, i]) for i in range(2)]) / l['time'],
]

plt.figure()
plt.boxplot(ess_per_sec, vert=False)
plt.grid(linestyle=':')
plt.yticks([1, 2, 3], ['Haario\n(R.W.M.)', 'Pseudo-Likelihood\n(I.M.H.)', 'Langevin'], fontsize=20)
plt.xlabel('Effective Sample Size per Second', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join('images', 'neal-funnel-ess-per-sec.png'))
