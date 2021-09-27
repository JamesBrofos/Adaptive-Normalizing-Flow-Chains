import argparse
import io
import os
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
import optax
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm

from distribution import distributions
from normalizing_flow import NormalizingFlow

parser = argparse.ArgumentParser(description='Markov chain Monte Carlo with normalizing flows in two-dimensional distributions')
parser.add_argument('--target', type=str, default='banana', help='Indicator of which target distribution to use')
parser.add_argument('--step-size', type=float, default=1e-4, help='Gradient descent learning rate')
parser.add_argument('--num-hidden', type=int, default=128, help='Number of hidden units used in the neural networks')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()
identfier = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items()).replace('_', '-')

distr = distributions[args.target]
sample, log_prob = NormalizingFlow(2, args.num_hidden)
optimizer = optax.adam(args.step_size)

def plot_to_image(figure: plt.Figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    Code modified from [1].

    [1] https://www.tensorflow.org/tensorboard/image_summaries

    Args:
        figure: A matplotlib figure.

    Returns:
        image: The rasterized graphic image of the figure in PNG format.

    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TensorFlow image.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension.
    image = tf.expand_dims(image, 0)
    return image

def visualize_samples(samples: jnp.ndarray) -> plt.Figure:
    """Visualizes the samples.

    Args:
        iid: Independent analytical samples.

    Returns:
        fig: Visualization of samples.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
    ax.grid(linestyle=':')
    ax.set_xlim((-3.0, 3.0))
    ax.set_ylim((-3.0, 3.0))
    fig.tight_layout()
    return fig

def visualize_ks_statistic(ks: list) -> plt.Figure:
    """Visualizes the Kolmogorov-Smirnov test statistic for comparing adaptive
    samples to analytical samples from the target distribution.

    Args:
        ks: List of Kolmogorov-Smirnov statistics along random directions.

    Returns:
        fig: Visualization of the Kolmogorov-Smirnov statistcs.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(ks, vert=False, notch=True)
    ax.grid(linestyle=':')
    fig.tight_layout()
    return fig

@jax.jit
def metropolis_hastings(
        params: hk.Params,
        rng: jax.random.PRNGKey,
        x: jnp.ndarray,
        h: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies the Metropolis-Hastings criterion to samples generated from the
    normalizing flow proposal distribution.

    Args:
        params: Parameters of the normalizing flow.
        rng: Pseudo-random number generator key.
        x: The current state of the chain.
        h: The historical accepted samples of the chain.

    Returns:
        x: The updated state of the chain.
        h: The updated historical accepted samples of the chain.
        accprob: The acceptance probability at this step of the chain.

    """
    mix = 0.1
    rng_z, rng_u, rng_m, rng_mix = jax.random.split(rng, 4)
    num_samples = len(x)
    zp = sample.apply(params, rng_z, num_samples)
    zm = jax.random.normal(rng_m, zp.shape)
    a = jnp.atleast_2d(jax.random.uniform(rng_mix, [num_samples]) < mix).T
    z = jnp.where(a, zm, zp)
    pz = jnp.stack([jspst.norm.logpdf(z).sum(axis=-1), log_prob.apply(params, z)], axis=-1)
    px = jnp.stack([jspst.norm.logpdf(x).sum(axis=-1), log_prob.apply(params, x)], axis=-1)
    lq_z = jspsp.logsumexp(pz, axis=-1, b=jnp.array([mix, 1-mix]))
    lq_x = jspsp.logsumexp(px, axis=-1, b=jnp.array([mix, 1-mix]))
    lp_z = distr.log_prob(z)
    lp_x = distr.log_prob(x)
    mh = lp_z - lp_x + lq_x - lq_z
    log_u = jnp.log(jax.random.uniform(rng_u, [num_samples]))
    accept = jnp.atleast_2d(log_u < mh).T
    accprob = jnp.mean(accept)
    h = jnp.where(accept, x, h)
    x = jnp.where(accept, z, x)
    return x, h, accprob

def zero_nans(mat: jnp.ndarray) -> jnp.ndarray:
    """Removes the NaNs in a matrix by replacing them with zeros. This can be used
    to avert unstable training.

    Args:
        mat: Matrix whose NaN entries should be replaced by zero.

    Returns:
        out: The input with NaNs replaced by zeros.

    """
    out = jnp.where(jnp.isnan(mat), 0.0, mat)
    return out

loss = lambda params, x: -jnp.mean(log_prob.apply(params, x))

@jax.jit
def update(
        params: hk.Params,
        rng: jax.random.PRNGKey,
        opt_state: optax.OptState,
        samples: jnp.ndarray,
) -> Tuple[hk.Params, optax.OptState]:
    """Applies a gradient descent update to the parameters of a RealNVP normalizing
    flow architecture trained using a Monte Carlo approximation to the KL
    divergence.

    Args:
        params: Parameters of the normalizing flow.
        rng: Pseudo-random number generator key.
        opt_state: The current state of the optimizer.
        samples: Samples drawn approximately from the target distribution to
            compute the loss function.

    Returns:
        new_params: The updated parameters of the normalizing flow.
        new_opt_state: The updated state of the optimizer.
        l: The value of the loss function given the current parameters.

    """
    l, grad = jax.value_and_grad(loss)(params, samples)
    grad = jax.tree_util.tree_map(zero_nans, grad)
    updates, new_opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, l


def main():
    import scipy.stats as spst

    hkrng = hk.PRNGSequence(args.seed)

    params = log_prob.init(next(hkrng), jnp.zeros([2]))
    opt_state = optimizer.init(params)

    logdir = os.path.join('logs', identfier)
    writer =  tf.summary.create_file_writer(logdir)

    iid = jax.random.normal(next(hkrng), [100, 2])
    h = jnp.asarray(iid)
    x = jnp.asarray(iid)
    xs = []

    for i in tqdm.tqdm(range(100000)):
        step = i + 1
        x, h, ap = metropolis_hastings(params, next(hkrng), x, h)
        xs.append(x)
        params, opt_state, l = update(params, next(hkrng), opt_state, h)
        with writer.as_default():
            if step % 100 == 0:
                tf.summary.scalar('loss', l, step=step)
                tf.summary.scalar('acc prob', ap, step=step)
            if step % 10000 == 0:
                s = sample.apply(params, next(hkrng), len(x))
                fig = visualize_samples(s)
                im = plot_to_image(fig)
                tf.summary.image('proposal', im, step=step)
                fig = visualize_samples(x)
                im = plot_to_image(fig)
                tf.summary.image('mcmc samples', im, step=step)

    with writer.as_default():
        xs = jnp.reshape(jnp.array(xs), [-1, 2])
        fig = visualize_samples(xs)
        im = plot_to_image(fig)
        tf.summary.image('all samples', im, step=step)

    ys = distr.sample(len(xs), seed=next(hkrng))
    u = jax.random.normal(next(hkrng), [100, 2])
    u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
    ysu = ys@u.T
    xsu = xs@u.T
    ks = []
    for i in tqdm.tqdm(range(len(u))):
        stat = spst.ks_2samp(ysu[:, i], xsu[1000000:, i]).statistic
        ks.append(stat)

    with writer.as_default():
        fig = visualize_ks_statistic(ks)
        im = plot_to_image(fig)
        tf.summary.image('kolmogorov smirnov', im, step=step)

if __name__ == '__main__':
    main()
