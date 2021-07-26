import argparse
import io
import os
import random
from typing import Callable, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import scipy.stats as spst
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
import tqdm

tfj = tfp.tf2jax
tfb = tfp.bijectors
tfd = tfp.distributions

import banana
import multimodal
import neal


parser = argparse.ArgumentParser(description='Markov chain Monte Carlo with normalizing flows')
parser.add_argument('--target', type=str, default='banana', help='Indicator of which target distribution to use')
parser.add_argument('--num-mc', type=int, default=1, help='Number of samples to use in the Monte Carlo estimate of the loss')
parser.add_argument('--reg-klqp', type=float, default=1.0, help='Weight multiplier on reverse KL term in loss')
parser.add_argument('--reg-klpq', type=float, default=1.0, help='Weight multiplier on forward KL term in loss')
parser.add_argument('--reg-ap', type=float, default=1.0, help='Weight multiplier on acceptance probability term in loss')
parser.add_argument('--step-size', type=float, default=1e-4, help='Gradient descent learning rate')
parser.add_argument('--num-steps', type=int, default=50000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--num-hidden', type=int, default=70, help='Number of hidden units used in the neural networks')
parser.add_argument('--num-realnvp', type=int, default=3, help='Number of RealNVP bijectors to employ')
parser.add_argument('--temp', type=float, default=1.0, help='Target distribution temperature parameter')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

_id = '-'.join('{}-{}'.format(k, v) for k, v in vars(args).items()).replace('_', '-')

target = {
    'banana': banana,
    'multimodal': multimodal,
    'neal': neal
}[args.target]


class ShiftAndLogScale(hk.Module):
    """Parameterizes the shift and scale functions used in the RealNVP normalizing
    flow architecture.

    Args:
        num_hidden: Number of hidden units in each hidden layer.

    """
    def __init__(self, num_hidden: int):
        super().__init__()
        self._num_hidden = num_hidden

    def __call__(
            self,
            x: jnp.ndarray,
            input_depth: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = hk.Flatten()(x)
        x = hk.Linear(self._num_hidden)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_hidden)(x)
        x = jax.nn.relu(x)
        shift = hk.Linear(input_depth)(x)
        log_scale = hk.Linear(input_depth)(x)
        return shift, log_scale

def build_bijector(num_dims: int, num_hidden: int) -> tfb.Bijector:
    """Constructs a bijector object from the composition of RealNVP layers and
    permutation layers.

    Args:
        num_dims: The dimensionality of the input to the bijector.
        num_hidden: The number of hidden units used in the RealNVP bijectors.

    Returns:
        bij: The bijector composed of the concatenation of RealNVP and
            permutation bijectors.

    """
    num_masked = num_dims // 2
    bij = tfb.Chain([
        tfb.RealNVP(num_masked=num_masked,
                    shift_and_log_scale_fn=ShiftAndLogScale(num_hidden)),
        tfb.Permute(list(reversed(range(num_dims)))),
        tfb.RealNVP(num_masked=num_masked,
                    shift_and_log_scale_fn=ShiftAndLogScale(num_hidden)),
        tfb.Permute(list(reversed(range(num_dims)))),
        tfb.RealNVP(num_masked=num_masked,
                    shift_and_log_scale_fn=ShiftAndLogScale(num_hidden)),
        tfb.Permute(list(reversed(range(num_dims)))),
        tfb.RealNVP(num_masked=num_masked,
                    shift_and_log_scale_fn=ShiftAndLogScale(num_hidden)),
    ])
    return bij

def build_distribution(num_dims: int, num_hidden: int) -> tfd.TransformedDistribution:
    """Constructs the distribution associated to applying the bijector to a base
    distribution, which is a standard multivariate Gaussian.

    Args:
        num_dims: The dimensionality of the input to the bijector.
        num_hidden: The number of hidden units used in the RealNVP bijectors.

    Returns:
        dist: The distribution produced by applying the bijector to the base
            distribution.

    """
    dist = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(
            loc=jnp.zeros(num_dims),
            scale_diag=jnp.ones(num_dims)
        ),
        bijector=build_bijector(num_dims, num_hidden)
    )
    return dist

def NormalizingFlow(num_dims: int, num_hidden: int) -> Tuple[hk.Transformed, hk.Transformed]:
    """Constructs the sampling and log-density functions of the normalizing flow,
    which is composed of RealNVP bijectors and permutation layers.

    Args:
        num_dims: The dimensionality of the input to the bijector.
        num_hidden: The number of hidden units used in the RealNVP bijectors.

    Returns:
        sample: Transformed object with method to sample from the normalizing
            flow.
        log_prob: Transformed object with method to compute the log-density of
            the normalizing flow.

    """
    @hk.transform
    def sample(*args, **kwargs):
        return build_distribution(num_dims, num_hidden).sample(seed=hk.next_rng_key(), *args, **kwargs)

    @hk.without_apply_rng
    @hk.transform
    def log_prob(*args, **kwargs):
        return build_distribution(num_dims, num_hidden).log_prob(*args, **kwargs)

    return sample, log_prob

sample, log_prob = NormalizingFlow(2, args.num_hidden)


class State(NamedTuple):
    """Object to store the state of the Markov chain. This class is used to
    represent both the proposal and the current state of the chain.

    Parameters:
        state: State of the Markov chain.
        log_appox: The log-density of the proposal under the approximate
            distribution.
        log_target: The log-density of the target distribution evaluated at the
            proposal location.

    """
    state: jnp.ndarray
    log_approx: float
    log_target: float

def loss(
        params: hk.Params,
        rng: jax.random.PRNGKey,
        num_samples: int,
        temp: float,
        approx: State,
        reg_klqp: float,
        reg_klpq: float,
        reg_ap: float
) -> float:
    """Computes the loss function used to estimate the parameters of the
    normalizing flow. Includes the KL divergence between the approximate
    distribution and the target distribution wherein the expectation is
    approximated via a Monte Carlo sample. The target distribution can be
    modulated via a temperature parameter in order to induce a flattened
    distribution.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        rng: Pseudo-random number generator key.
        num_samples: Number of samples used in the Monte Carlo approximation of
            the expectation.
        temp: Temperature parameter to apply to the target distribution.
        approx: An approximate sample of the target distribution of the Markov
            chain.
        reg_klqp: Weight to apply to the reverse KL term in the loss function.
        reg_klpq: Weight to apply to the forward KL term in the loss function.
        reg_ap: Weight to apply to the acceptance probability term in the loss
            function.

    Returns:
        lv: The value of the loss function for estimating the parameters of the
            normalizing flow.
        prop: The proposal state of the Markov chain.

    """
    x = sample.apply(params, rng, num_samples)
    log_approx = log_prob.apply(params, x)
    log_target = target.log_prob(x)
    prop = State(x[0], log_approx[0], log_target[0])
    approx = State(approx.state,
                   log_prob.apply(params, approx.state),
                   approx.log_target)
    ent = -jnp.mean(log_approx)
    klqp = -jnp.mean(log_target / temp) - ent
    klpq = -jnp.mean(log_prob.apply(params, approx.state))
    ap = -metro(approx, prop)
    lv = reg_klqp*klqp + reg_klpq*klpq + reg_ap*ap
    return lv, prop

optimizer = optax.adam(args.step_size)

class ChainInfo(NamedTuple):
    """Object to store diagnostic information from a step of the chain.

    Args:
        state: The state of the chain.
        accepted: Whether or not the proposal at this step of the chain was
            accepted.
        loss_value: The value of the loss function used to train the normalizing
            flow.
        adapt_prob: The probability of adapting the parameters of the chain to
            guarantee diminishing adaptation.

    """
    state: State
    accepted: bool
    loss_value: float
    adapt_prob: float

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

@jax.partial(jax.jit, static_argnums=(4, ))
def update(
        it: int,
        params: hk.Params,
        rng: jax.random.PRNGKey,
        opt_state: optax.OptState,
        num_samples: int,
        temp: float,
        curr: State,
        approx: State,
        reg_klqp: float,
        reg_klpq: float,
        reg_ap: float
) -> Tuple:
    """Applies a gradient descent update to the parameters of a RealNVP normalizing
    flow architecture trained using a Monte Carlo approximation to the KL
    divergence. One of the samples used to compute the gradient of the KL
    divergence is then taken as the proposal state of an adaptive chain, and it
    is accepted or rejected according to the Metropolis-Hastings algorithm. If
    certain conditions on the adaptivity of the proposal distribution, and on
    the suitability of the proposal distribution for sampling the target, are
    met, then this represents an ergodic procedure.

    Args:
        it: Iteration counter.
        params: Parameters of the RealNVP bijector.
        rng: Pseudo-random number generator key.
        opt_state: The current state of the optimizer.
        temp: Temperature parameter to apply to the target distribution.
        curr: The current state of the chain.
        approx: An approximate sample of the target distribution of the Markov
            chain.
        reg_klqp: Weight to apply to the reverse KL term in the loss function.
        reg_klpq: Weight to apply to the forward KL term in the loss function.
        reg_ap: Weight to apply to the acceptance probability term in the loss
            function.

    Returns:
        next_params: The updated parameters of the normalizing flow.
        next_opt_state: The updated state of the optimizer.
        info: Diagnostic information from a single step of the chain.

    """
    rng_prop, rng_metro, rng_da = jax.random.split(rng, 3)
    # Compute proposal using current parameters.
    (lossval, prop), grad = jax.value_and_grad(
        loss,
        has_aux=True)(params,
                      rng_prop,
                      num_samples,
                      temp,
                      approx,
                      reg_klqp,
                      reg_klpq,
                      reg_ap)
    curr = State(
        curr.state,
        log_prob.apply(params, curr.state),
        curr.log_target
    )
    curr, accept = accept_reject(rng_metro, prop, curr)
    # Modify parameters with diminishing adaptation guaranteed.
    grad = jax.tree_util.tree_map(zero_nans, grad)
    updates, new_opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    adapt_prob = 1.0 / (it + 1)**(1.0 / 10)
    next_params, next_opt_state = jax.lax.cond(
        jax.random.uniform(rng_da) < adapt_prob,
        lambda _: (new_params, new_opt_state),
        lambda _: (params, opt_state),
        None
    )
    info = ChainInfo(curr, accept, lossval, adapt_prob)
    return next_params, next_opt_state, info

def metro(prop: State, curr: State) -> float:
    """Computes the Metropolis-Hastings acceptance probability.

    Args:
        prop: The proposal for the next state of the chain.
        curr: The current state of the chain.

    Returns:
        m: The log-acceptance probability of the proposal given the current state
            of the chain.

    """
    m = prop.log_target - curr.log_target + curr.log_approx - prop.log_approx
    m = jnp.minimum(m, 0.0)
    return m

def accept_reject(
        rng: jax.random.PRNGKey,
        prop: State,
        curr: State,
) -> Tuple:
    """Applies the Metropolis-Hastings accept-reject criterion given the current
    state of the chain, the log-density of the target density at the current
    state, the log-density of the approximate distribution at the current
    state, and the corresponding quantities of the proposal.

    Args:
        rng: Pseuod-random number generator key.
        prop: The proposal for the next state of the chain.
        curr: The current state of the chain.

    Returns:
        next_state: The next state of the chain as determined by the
            Metropolis-Hastings accept-reject criterion.
        accept: Whether or not the proposal was accepted.

    """
    logu = jnp.log(jax.random.uniform(rng))
    accept = logu < metro(prop, curr)
    next_state, accepted = jax.lax.cond(accept,
                                        lambda _: (prop, True),
                                        lambda _: (curr, False),
                                        None)
    return next_state, accepted

def plot_to_image(figure):
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

def normalizing_flow_samples(iid: jnp.ndarray, approx: jnp.ndarray):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.set_title('Independent Samples')
    ax.plot(iid[:, 0], iid[:, 1], '.')
    ax.grid(linestyle=':')
    ax.set_xlim((-5.0, 5.0))
    ax.set_ylim((-30.0, 4.0))
    ax = fig.add_subplot(122)
    ax.set_title('Normalizing Flow Samples')
    ax.plot(approx[:, 0], approx[:, 1], '.')
    ax.grid(linestyle=':')
    ax.set_xlim((-5.0, 5.0))
    ax.set_ylim((-30.0, 4.0))
    fig.tight_layout()
    return fig

def normalizing_flow_density(params: hk.Params):
    xr = jnp.linspace(-5.0, 5.0, 500)
    yr = jnp.linspace(-30.0, 4.0, 500)
    xx, yy = jnp.meshgrid(xr, yr)
    grid = jnp.vstack([jnp.ravel(xx), jnp.ravel(yy)]).T
    tlp = target.log_prob(grid)
    prob = jnp.exp(tlp)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.contour(xx, yy, jnp.reshape(prob, xx.shape), cmap=plt.cm.Blues)
    ax.set_title('Analytical Density')
    ax = fig.add_subplot(122)
    alp = log_prob.apply(params, grid)
    prob = jnp.exp(alp)
    ax.contour(xx, yy, jnp.reshape(prob, xx.shape), cmap=plt.cm.Blues)
    ax.set_title('Normalizing Flow')
    fig.tight_layout()
    return fig

def adaptive_sample(iid: jnp.ndarray, adapt: jnp.ndarray):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.set_title('Independent Samples')
    ax.plot(iid[:, 0], iid[:, 1], '.')
    ax.grid(linestyle=':')
    ax.set_xlim((-5.0, 5.0))
    ax.set_ylim((-30.0, 4.0))
    ax = fig.add_subplot(122)
    ax.set_title('Adaptive Samples')
    ax.plot(adapt[:, 0], adapt[:, 1], '.')
    ax.grid(linestyle=':')
    ax.set_xlim((-5.0, 5.0))
    ax.set_ylim((-30.0, 4.0))
    fig.tight_layout()
    return fig

def kolmogorov_smirnov(iid: jnp.ndarray, adapt: jnp.ndarray):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121)
    ks = spst.ks_2samp(adapt[:, 0], iid[:, 0]).statistic
    ax.set_title('KS-Test Statistic: {:.4f}'.format(ks))
    ax.hist(adapt[:, 0], bins=100, density=True, histtype='step', cumulative=True, label='Adaptive Samples')
    ax.hist(iid[:, 0], bins=100, density=True, histtype='step', cumulative=True, label='Independent Samples')
    ax.legend()
    ax = fig.add_subplot(122)
    ks = spst.ks_2samp(adapt[:, 1], iid[:, 1]).statistic
    ax.set_title('KS-Test Statistic: {:.4f}'.format(ks))
    ax.hist(adapt[:, 1], bins=100, density=True, histtype='step', cumulative=True, label='Adaptive Samples')
    ax.hist(iid[:, 1], bins=100, density=True, histtype='step', cumulative=True, label='Independent Samples')
    fig.tight_layout()
    return fig


def main():
    random.seed(args.seed)
    hkrng = hk.PRNGSequence(args.seed)

    params = log_prob.init(next(hkrng), jnp.zeros([2]))

    curr = sample.apply(params, next(hkrng))
    lp = log_prob.apply(params, curr)
    log_target_curr = target.log_prob(curr)
    curr = State(curr, lp, log_target_curr)

    logdir = os.path.join('logs', _id)
    writer =  tf.summary.create_file_writer(logdir)

    iid = target.sample(args.num_steps, next(hkrng))
    adapt = [curr]
    opt_state = optimizer.init(params)

    for i in tqdm.tqdm(range(args.num_steps)):
        step = i + 1
        params, opt_state, info = update(
            i,
            params,
            next(hkrng),
            opt_state,
            args.num_mc,
            args.temp,
            curr,
            random.choice(adapt),
            args.reg_klqp,
            args.reg_klpq,
            args.reg_ap
        )
        curr = info.state
        adapt.append(curr)

        with writer.as_default():
            if step % 100 == 0:
                tf.summary.scalar('loss', info.loss_value, step=step)
                tf.summary.scalar('acc. prob.', info.accepted, step=step)
                tf.summary.scalar('adapt prob.', info.adapt_prob, step=step)
            if step % (args.num_steps) == 0:
                A = jnp.asarray([s.state for s in adapt])
                fig = normalizing_flow_samples(iid, sample.apply(params, next(hkrng), args.num_steps))
                im = plot_to_image(fig)
                tf.summary.image('normalizing flow samples', im, step=step)
                fig = normalizing_flow_density(params)
                im = plot_to_image(fig)
                tf.summary.image('normalizing flow density', im, step=step)
                fig = adaptive_sample(iid, A)
                im = plot_to_image(fig)
                tf.summary.image('adaptive samples', im, step=step)
                fig = kolmogorov_smirnov(iid, A)
                im = plot_to_image(fig)
                tf.summary.image('kolmogorov smirnov', im, step=step)

if __name__ == '__main__':
    main()
