import jax
import jax.numpy as jnp
import pandas as pd
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


def hogg():
    dfhogg = pd.DataFrame(jnp.array([[1, 201, 592, 61, 9, -0.84],
                                     [2, 244, 401, 25, 4, 0.31],
                                     [3, 47, 583, 38, 11, 0.64],
                                     [4, 287, 402, 15, 7, -0.27],
                                     [5, 203, 495, 21, 5, -0.33],
                                     [6, 58, 173, 15, 9, 0.67],
                                     [7, 210, 479, 27, 4, -0.02],
                                     [8, 202, 504, 14, 4, -0.05],
                                     [9, 198, 510, 30, 11, -0.84],
                                     [10, 158, 416, 16, 7, -0.69],
                                     [11, 165, 393, 14, 5, 0.30],
                                     [12, 201, 442, 25, 5, -0.46],
                                     [13, 157, 317, 52, 5, -0.03],
                                     [14, 131, 311, 16, 6, 0.50],
                                     [15, 166, 400, 34, 6, 0.73],
                                     [16, 160, 337, 31, 5, -0.52],
                                     [17, 186, 423, 42, 9, 0.90],
                                     [18, 125, 334, 26, 8, 0.40],
                                     [19, 218, 533, 16, 6, -0.78],
                                     [20, 146, 344, 22, 5, -0.56]]),
                          columns=['id','x','y','sigma_y','sigma_x','rho_xy'])

    # for convenience zero-base the 'id' and use as index
    dfhogg['id'] = dfhogg['id'] - 1
    dfhogg.set_index('id', inplace=True)

    # standardize (mean center and divide by 1 sd)
    dfhoggs = (dfhogg[['x','y']] - dfhogg[['x','y']].mean(0)) / dfhogg[['x','y']].std(0)
    dfhoggs['sigma_y'] = dfhogg['sigma_y'] / dfhogg['y'].std(0)
    dfhoggs['sigma_x'] = dfhogg['sigma_x'] / dfhogg['x'].std(0)

    X = jnp.asarray(dfhoggs['x'].values)
    sigma_y = jnp.asarray(dfhoggs['sigma_y'].values)
    y = jnp.asarray(dfhoggs['y'].values)
    return X, y, sigma_y

num_dims = 2
X, y, sigma_y = hogg()

dist = tfd.JointDistributionSequential([
    # b0
    tfd.Normal(loc=0.0, scale=1.0),
    # b1
    tfd.Normal(loc=0.0, scale=1.0),
    # likelihood
    #   Using Independent to ensure the log_prob is not incorrectly broadcasted
    lambda b1, b0: tfd.Independent(
        tfd.Normal(
            # Parameter transformation
            # b1 shape: (batch_shape), X shape (num_obs): we want result to have
            # shape (batch_shape, num_obs)
            loc=b0 + b1*X,
            scale=sigma_y),
        reinterpreted_batch_ndims=1
    ),
])

def log_prob(x):
    lp = dist.log_prob([x[..., 0], x[..., 1], y])
    return lp

@jax.jit
def run_chain(init_state, step_size, unconstraining_bijectors,
              num_steps=1000000, burnin=1000):

    def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio
        )

    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.NoUTurnSampler(
            log_prob,
            step_size=step_size
        ),
        bijector=unconstraining_bijectors)

    hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=burnin,
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio
    )

    # Sampling from the chain.
    chain_state, sampler_stat = tfp.mcmc.sample_chain(
        num_results=num_steps,
        num_burnin_steps=burnin,
        current_state=init_state,
        kernel=hmc,
        trace_fn=trace_fn,
        seed=jax.random.PRNGKey(0))
    return chain_state, sampler_stat

def main():
    import pickle
    import numpy as onp

    xa, xb, _ = dist.sample(seed=jax.random.PRNGKey(0))
    x = jnp.hstack([xa, xb])
    step_size = 0.1
    samples, sampler_stat = run_chain(x, step_size, tfb.Identity())

    with open('hogg-samples.pkl', 'wb') as f:
        pickle.dump(onp.asarray(samples), f)

if __name__ == '__main__':
    main()
