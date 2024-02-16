from argparse import ArgumentParser
import json
from load import load_posteriors
from utils import nf_to_source
from flows import init_nf_catalog
import equinox as eqx
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from corner import corner
import numpy as np
import numpyro
import numpyro.distributions as dists
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import TransformReparam
from transforms import FlowMCTransform
import arviz as az


def Args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='/projects/farr_lab/shared/GWTC3/all_events')
    parser.add_argument("--model-dir", type=str, default='./trained')
    parser.add_argument("--outdir", type=str, default='./trained/corner_plots')
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    return parser.parse_args()


def model(N, nparams, catalog_transform):
    # Sample the catalog
    with numpyro.handlers.reparam(config={'catalog': TransformReparam()}):
        base_dist = dists.Normal(jnp.zeros((N, nparams)), jnp.ones((N, nparams)))
        catalog = numpyro.sample(
            "catalog",
            dists.TransformedDistribution(base_dist,
                                          catalog_transform),
        )
    m1, q, a1, a2, tilt1, tilt2, z = nf_to_source(catalog)
    numpyro.deterministic('mass_1', m1)
    numpyro.deterministic('mass_ratio', q)
    numpyro.deterministic('a_1', a1)
    numpyro.deterministic('a_2', a2)
    numpyro.deterministic('cos_tilt_1', tilt1)
    numpyro.deterministic('cos_tilt_2', tilt2)
    numpyro.deterministic('redshift', z)


def main():
    args = Args()

    param_names = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']

    # Load trained normalizing flow catalog
    with open(f"{args.model_dir}/events.json", "r") as inp:
        nf_events = json.load(inp)

    nevents = len(nf_events)
    nparams = 7
    key = random.PRNGKey(150914)
    key, init_key, MCMCKey = random.split(key, 3)
    init_keys = random.split(init_key, nevents)
    keys = random.split(key, nevents)

    nf_cat = init_nf_catalog(keys, None, None)
    nf_cat = eqx.tree_deserialise_leaves(f"{args.model_dir}/trained_NF_catalog.eqx", nf_cat)

    nf_transformed_draws = nf_cat.sample_ensemble(keys, args.num_samples).transpose(1, 0, 2)
    m1, q, a1, a2, tilt1, tilt2, z = eqx.filter_vmap(nf_to_source)(nf_transformed_draws)
    nf_draws = np.array([m1, q, a1, a2, tilt1, tilt2, z]).T

    # Load posterior samples
    posts, post_events = load_posteriors(args.data_dir, params=param_names+['prior'], use_chirp_mass=False)
    pe_samples = {e: np.array([posts[e][p] for p in param_names]).T for e in post_events}
    pe_priors = {e: posts[e]['prior'] for e in post_events}

    # sample to test tranrform and log_prob
    catalog_transform = FlowMCTransform(nf_cat)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_chains=args.num_chains, num_warmup=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(MCMCKey, nevents, nparams, catalog_transform, extra_fields=('potential_energy',))
    idata = az.from_numpyro(mcmc)
    
    label = f'{args.outdir}/catalog'

    fig = az.plot_trace(idata)
    plt.tight_layout()
    plt.savefig(f'{label}_traceplot.png')
    plt.close()
    
    fig = plt.figure()
    pe = mcmc.get_extra_fields()['potential_energy']
    plt.plot(pe)
    plt.savefig(f'{label}_pe.png')
    plt.close()

    posterior_samples = mcmc.get_samples()
    catalog_samples = np.array([posterior_samples[p] for p in param_names]).T

    for nf_e, nf_event in enumerate(nf_events):
        print(f"plotting {nf_event}")
        fig = corner(nf_draws[nf_e], labels=param_names, hist_kwargs={'density': True}, color='tab:blue')
        corner(catalog_samples[nf_e], fig=fig, hist_kwargs={'density': True}, color='tab:purple')
        corner(pe_samples[nf_event], weights=1/pe_priors[nf_event], fig=fig, hist_kwargs={'density': True}, color='tab:gray')
        plt.suptitle(nf_event, fontsize='xx-large')
        fig.savefig(f"{args.outdir}/{nf_event}_corner.png")
        plt.close(fig)


if __name__ == "__main__":
    main()