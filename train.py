from argparse import ArgumentParser
import json
from load import load_posteriors
from utils import remove_bounds, compute_cat_mean_cov
from flows import init_nf_catalog
import optax
import equinox as eqx
from typing import Callable, Tuple
import jax
from jax import random
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import matplotlib.pyplot as plt
from gwinferno.preprocess.data_collection import p_m1src_q_z_lal_pe_prior

# TODO: SOURCE FRAME CHIRP MASS
def Args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='/projects/farr_lab/shared/GWTC3/all_events')
    parser.add_argument("--outdir", type=str, default='./trained')
    parser.add_argument("--epochs", type=int, default=250)
    return parser.parse_args()


def make_training_loop(optim: optax.GradientTransformation) -> Callable:
    """
    Create a function that trains an NF model.

    Args:
        model (eqx.Model): NF model to train.
        optim (optax.GradientTransformation): Optimizer.

    Returns:
        train_flow: Function that trains the model.
    """

    @eqx.filter_value_and_grad
    def loss_fn(model, x, w):
        return -jnp.mean(w * model.log_prob(x))

    @eqx.filter_jit
    def train_step(model, x, w, opt_state):
        """Train for a single step.

        Args:
            model (eqx.Model): NF model to train.
            x (Array): Training data.
            w (Array): Training weights.
            opt_state (optax.OptState): Optimizer state.

        Returns:
            loss (Array): Loss value.
            model (eqx.Model): Updated model.
            opt_state (optax.OptState): Updated optimizer state.
        """
        loss, grads = loss_fn(model, x, w)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def train_epoch(rng, model, state, train_ds, train_ws, batch_size):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)

            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = train_ds[perm, ...]
                batch_ws = train_ws[perm, ...]
                value, model, state = train_step(model, batch, batch_ws, state)
        else:
            value, model, state = train_step(model, train_ds, train_ws, state)

        return value, model, state

    @eqx.filter_vmap
    def train_flow(
        rng: PRNGKeyArray,
        model: eqx.Module,
        data: Array,
        weights: Array,
        state: optax.OptState,
        num_epochs: int,
        batch_size: int,
        # verbose: bool = True,
    ) -> Tuple[PRNGKeyArray, eqx.Module, Array]:
        """Train a normalizing flow model.

        Args:
            rng (PRNGKeyArray): JAX PRNGKey.
            model (eqx.Module): NF model to train.
            data (Array): Training data.
            weights (Array): Training weights.
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            verbose (bool): Whether to print progress.

        Returns:
            rng (PRNGKeyArray): Updated JAX PRNGKey.
            model (eqx.Model): Updated NF model.
            loss_values (Array): Loss values.
        """
        loss_values = jnp.zeros(num_epochs)
        # if verbose:
        #     pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        # else:
        pbar = range(num_epochs)
        # best_model = model
        # best_loss = 1e9
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, model, state = train_epoch(input_rng, model, state, data, weights, batch_size)
            loss_values = loss_values.at[epoch].set(value)
            # if loss_values[epoch] < best_loss:
            #     best_model = model
            #     best_loss = loss_values[epoch]
            # if verbose:
            #     if num_epochs > 10:
            #         if epoch % int(num_epochs / 10) == 0:
            #             pbar.set_description(f"Training NF, current loss: {value:.3f}")
            #     else:
            #         if epoch == num_epochs:
            #             pbar.set_description(f"Training NF, current loss: {value:.3f}")

        return rng, model, state, loss_values

    return train_flow, train_epoch, train_step


def main():
    args = Args()
    posts, events = load_posteriors(args.data_dir)

    param_names = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']
    param_bounds = {'chirp_mass': (0, None), 'mass_ratio': (0, 1), 'a_1': (0, 1), 'a_2': (0, 1), 'cos_tilt_1': (-1, 1), 'cos_tilt_2': (-1, 1), 'redshift': (0, None)}
    param_names_transformed = [f"{p}_logit" if None not in param_bounds[p] else p for p in param_names]
    posts = {e: remove_bounds(posts[e], param_bounds) for e in events}

    nevents = len(events)
    ndim = len(param_names_transformed)
    key = random.PRNGKey(150914)
    keys = random.split(key, nevents)

    # Prepare data for training and initialize catalog
    data = jnp.array([jnp.array([jnp.array(post[p]) for p in param_names_transformed]).T for post in posts.values()])
    data_mean, data_cov = compute_cat_mean_cov(data)
    weights = 1/jnp.array([post['prior'] for post in posts.values()])
    print(weights.shape)

    nf_cat = init_nf_catalog(keys, data_mean, data_cov)  # TODO: make ndim flexible

    # Train catalog
    learning_rate = 0.002
    momentum = 0.8
    opt = optax.adam(learning_rate, momentum)

    @eqx.filter_vmap
    def init_optimizer_state(model):
        return opt.init(eqx.filter(model, eqx.is_array))

    states = init_optimizer_state(nf_cat)

    train_flows, train_epoch, train_step = make_training_loop(opt)

    batch_size = data.shape[1]
    rng, model, state, loss_vals = train_flows(keys, nf_cat, data, weights, states, args.epochs, batch_size)
    plt.plot(loss_vals.T)
    plt.savefig(f"{args.outdir}/loss.png")

    # Save trained model
    with open(f"{args.outdir}/events.json", "w") as outp:
        json.dump(events, outp)
    eqx.tree_serialise_leaves(f"{args.outdir}/trained_NF_catalog.eqx", model)


if __name__ == "__main__":
    main()
