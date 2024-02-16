from typing import Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline


class EnsembledMaskedCouplingRQSpline(MaskedCouplingRQSpline):
    @eqx.filter_vmap
    def log_prob_ensemble(self, x: Array) -> Array:
        """ From data space to latent space"""
        x = (x-self.data_mean)/jnp.sqrt(jnp.diag(self.data_cov)) # non-enembled transforms don't standardize
        y, log_det = self.__call__(x)
        log_det = log_det + self.base_dist.log_prob(y)
        return log_det

    @eqx.filter_vmap
    def inverse_ensemble(self, x: Array) -> Tuple[Array, Array]:
        """ From latent space to data space"""
        log_det = 0.
        for layer in reversed(self.layers):
            x, log_det_i = layer.inverse(x)
            log_det += log_det_i
        x = x * jnp.sqrt(jnp.diag(self.data_cov)) + self.data_mean
        log_det += jnp.log(jnp.sqrt(jnp.diag(self.data_cov))).sum(-1)
        return x, log_det

    @eqx.filter_vmap
    def forward_ensemble(self, x: Array) -> Tuple[Array, Array]:
        log_det = 0.
        x = (x-self.data_mean)/jnp.sqrt(jnp.diag(self.data_cov))
        log_det += jnp.log(1/jnp.sqrt(jnp.diag(self.data_cov))).sum(-1)
        for layer in self.layers:
            x, log_det_i = layer(x)
            log_det += log_det_i
        return x, log_det

    @eqx.filter_vmap
    def sample_ensemble(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        samples = self.base_dist.sample(rng_key, n_samples)
        samples = self.inverse(samples)[0]
        samples = samples * jnp.sqrt(jnp.diag(self.data_cov)) + self.data_mean # non-enembled transforms don't standardize
        return samples


@eqx.filter_vmap
def init_nf_catalog(key, data_mean, data_cov):
    ndim = 7
    n_layers = 8
    n_hidden = 12
    n_bins = 10
    return EnsembledMaskedCouplingRQSpline(
        ndim,
        n_layers,
        [n_hidden, n_hidden],
        n_bins,
        data_mean=data_mean,
        data_cov=data_cov,
        key=key,
    )