import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst


def get_chain(violate, num_steps):
    re = os.path.join('samples', 'samples-violate-{}-num-steps-{}-*.pkl'.format(violate, num_steps))
    fns = glob.glob(re)
    chain = []
    kl = []
    for fn in fns:
        with open(fn, 'rb') as f:
            dat = pickle.load(f)
            c = dat['chain']
            k = dat['klpq']
            chain.append(c)
            kl.append(k)
    chain = np.ravel(np.array(chain))
    kl = np.ravel(np.array(kl))
    return chain, kl


for violate in [True, False]:
    for num_steps in [10, 100, 1000, 10000]:
        chain, kl = get_chain(violate, num_steps)
        r = np.linspace(-3.0, 10.0, 1000)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'Violation of Stationarity', fontsize=20)
        ax.hist(chain, bins=50, density=True)
        ax.plot(r, spst.norm.pdf(r, 1.0, 0.5), '--', label='Target', linewidth=3)
        ax.grid(linestyle=':')
        ax.set_ylim((0.0, 1.0))
        fig.tight_layout()
        fig.savefig(os.path.join('images', 'adaptive-stationarity-violate-{}-num-steps-{}.png').format(violate, num_steps))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(r'$\log_{10}$ KL Divergence', fontsize=20)
ax.set_ylabel(r'Probability Density', fontsize=20)
chain, kl = get_chain(True, 100)
ax.hist(np.log10(kl), bins=50, density=True, alpha=0.8, label='100 Steps')
chain, kl = get_chain(True, 1000)
ax.hist(np.log10(kl), bins=50, density=True, alpha=0.8, label='1,000 Steps')
chain, kl = get_chain(True, 10000)
ax.hist(np.log10(kl), bins=50, density=True, alpha=0.8, label='10,000 Steps')
ax.grid(linestyle=':')
ax.set_xlim(-7, 6)
ax.set_ylim(0, 1)
ax.legend(fontsize=20)
fig.tight_layout()
fig.savefig(os.path.join('images', 'adaptive-stationarity-kl'))
