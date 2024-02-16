from jax import random
from jax.lax import broadcast_shapes
from jax.scipy.integrate import trapezoid
import jax.numpy as jnp
from numpyro.distributions import (
    Distribution,
    constraints,
    transforms,
)
from numpyro.distributions.util import (
    validate_sample,
    promote_shapes,
    is_prng_key,
)
import equinox as eqx
from gwinferno.models.bsplines.smoothing import apply_difference_prior
from utils import cumtrapz


class ProbabilisticCatalog(Distribution):
    """
    Utility numpyro distribution for sampling from a probabilistic catalog built of ensembled normalizing flows.
    """
    pytree_data_fields = ("emulator", "nevents", "nparams")

    def __init__(self, emulator, nevents, nparams, validate_args=None):
        self._support = constraints.real_vector
        self.emulator = emulator
        self.nevents = nevents
        self.nparams = nparams
        event_shape = (nevents, nparams)
        batch_shape = ()
        super(ProbabilisticCatalog, self).__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @eqx.filter_jit
    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        if jnp.size(sample_shape) > 1:
            raise NotImplementedError("Only 1D sample shapes are supported")
        n_samples = 1 if sample_shape == () else sample_shape[0]
        keys = random.split(key, self.nevents)
        draws = self.emulator.sample_ensemble(keys, n_samples)
        return jnp.squeeze(draws.transpose(1, 0, 2))
    
    def __call__(self, *args, **kwargs):
        key = kwargs.pop("rng_key")
        return self.sample(key, *args, **kwargs)

    @eqx.filter_jit
    def log_prob(self, value):
        return self.emulator.log_prob_ensemble(value).sum(axis=-1)


class Powerlaw(Distribution):
    arg_constraints = {
        "minimum": constraints.real,
        "maximum": constraints.real,
        "alpha": constraints.real,
    }
    reparametrized_params = ["minimum", "maximum", "alpha"]

    def __init__(self, alpha, minimum=0.0, maximum=1.0, low=0.0, high=1.0, validate_args=None):
        self.minimum, self.maximum, self.alpha = promote_shapes(minimum, maximum, alpha)
        self._support = constraints.interval(low, high)
        batch_shape = broadcast_shapes(
            jnp.shape(minimum),
            jnp.shape(maximum),
            jnp.shape(alpha),
        )
        super(Powerlaw, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value):
        logp = self.alpha * jnp.log(value)
        logp = logp + jnp.log((1.0 + self.alpha) / (self.maximum ** (1.0 + self.alpha) - self.minimum ** (1.0 + self.alpha)))
        logp_neg1 = -jnp.log(value) - jnp.log(self.maximum / self.minimum)
        return jnp.where(jnp.less(value, self.minimum) | jnp.greater(value, self.maximum), jnp.nan_to_num(-jnp.inf), jnp.where(jnp.equal(self.alpha, -1.0), logp_neg1, logp))

    def cdf(self, value):
        cdf = jnp.atleast_1d(value ** (self.alpha + 1.0) - self.minimum ** (self.alpha + 1.0)) / (
            self.maximum ** (self.alpha + 1.0) - self.minimum ** (self.alpha + 1.0)
        )
        cdf_neg1 = jnp.log(value / self.minimum) / jnp.log(self.maximum / self.minimum)
        cdf = jnp.where(jnp.equal(self.alpha, -1.0), cdf_neg1, cdf)
        cdf = jnp.minimum(cdf, 1.0)
        cdf = jnp.maximum(cdf, 0.0)
        return cdf

    def icdf(self, q):
        icdf = (self.minimum ** (1.0 + self.alpha) + q * (self.maximum ** (1.0 + self.alpha) - self.minimum ** (1.0 + self.alpha))) ** (
            1.0 / (1.0 + self.alpha)
        )
        icdf_neg1 = self.minimum * jnp.exp(q * jnp.log(self.maximum / self.minimum))
        return jnp.where(jnp.equal(self.alpha, -1.0), icdf_neg1, icdf)
    

class PowerlawRedshift(Distribution):
    arg_constraints = {
        "maximum": constraints.positive,
        "lamb": constraints.real,
    }
    reparametrized_params = ["maximum", "lamb"]

    def __init__(self, lamb, maximum, zgrid, dVcdz, low=0.0, high=1000.0, validate_args=None):
        self.maximum, self.lamb = promote_shapes(maximum, lamb)
        self._support = constraints.interval(low, high)
        batch_shape = broadcast_shapes(
            jnp.shape(maximum),
            jnp.shape(lamb),
        )
        super(PowerlawRedshift, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        self.zs = zgrid
        self.dVdc_ = dVcdz
        self.pdfs = self.dVdc_ * (1 + self.zs)**(lamb - 1)
        self.norm = trapezoid(self.pdfs, self.zs)
        self.pdfs /= self.norm
        self.cdfgrid = cumtrapz(self.pdfs, self.zs)
        self.cdfgrid = self.cdfgrid.at[-1].set(1)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape+self.batch_shape))

    @validate_sample
    def log_prob(self, value, dVdc=None):
        if dVdc is None:
            dVdc = jnp.interp(value, self.zs, self.dVdc_)
        return jnp.where(jnp.less_equal(value, self.maximum), jnp.log(dVdc) + (self.lamb - 1.0) * jnp.log(1.0 + value) - jnp.log(self.norm), jnp.nan_to_num(-jnp.inf))

    def cdf(self, value):
        return jnp.interp(value, self.zs, self.cdfgrid)

    def icdf(self, q):
        return jnp.interp(q, self.cdfgrid, self.zs)


class PowerlawSmoothedPowerlaw(Distribution):
    arg_constraints = {
        "minimum": constraints.positive,
        "maximum": constraints.positive,
        "alpha": constraints.real,
        "alpha_max": constraints.positive,
        "alpha_min": constraints.positive,
    }
    reparametrized_params = ["minimum", "maximum", "alpha", "alpha_max", "alpha_min"]

    def __init__(self, alpha, minimum, maximum, alpha_max, alpha_min, low, high, validate_args=None):
        self.minimum, self.maximum, self.alpha, self.alpha_max, self.alpha_min = promote_shapes(minimum, maximum, alpha, alpha_max, alpha_min)
        self.alpha_max = -self.alpha_max
        self._support = constraints.interval(low,high)
        self.low,self.high = low,high
        batch_shape = broadcast_shapes(
            jnp.shape(maximum), jnp.shape(minimum), jnp.shape(alpha), jnp.shape(alpha_max), jnp.shape(alpha_min)
        )
        super(PowerlawSmoothedPowerlaw, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        gamma = (self.alpha_min+1)/(self.minimum**(self.alpha_min+1)-self.low**(self.alpha_min+1))
        self.k1 = -gamma / (1+gamma/(self.alpha+1)*self.minimum**(self.alpha_min-self.alpha)*(self.minimum**(self.alpha+1)-self.maximum**(self.alpha+1))+gamma/(self.alpha_max+1)*self.minimum**(self.alpha_min-self.alpha)*self.maximum**(self.alpha-self.alpha_max)*(self.maximum**(self.alpha_max+1)-self.high**(self.alpha_max+1)))
        self.k2 = self.k1 * self.minimum**(self.alpha_min-self.alpha)
        self.k3 = self.k2 * self.maximum**(self.alpha-self.alpha_max)
    
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape=sample_shape + self.batch_shape
        return jnp.ones(shape)

    @validate_sample
    def log_prob(self, value):
        low_pl = jnp.where(jnp.less(value, self.minimum), jnp.log(self.k1) + jnp.log(value)*self.alpha_min, 0.0)
        high_pl = jnp.where(jnp.greater(value, self.maximum), jnp.log(self.k3) + jnp.log(value)*self.alpha_max, 0.0)
        mid_pl = jnp.where(jnp.greater_equal(value, self.minimum), jnp.where(jnp.less_equal(value, self.maximum), jnp.log(self.k2) + jnp.log(value)*self.alpha, 0.0), 0.0)
        return low_pl + mid_pl + high_pl
        #return jnp.where(jnp.isnan(pl) | jnp.isinf(pl), -jnp.nan_to_num(jnp.inf), pl)


# class BSplineDistribution(Distribution):
#     arg_constraints = {
#         "maximum": constraints.real,
#         "minimum": constraints.real,
#         "coefficients": constraints.real_vector,
#         "grid": constraints.real_vector,
#         "grid_dmat": constraints.real_matrix,
#     }
#     reparametrized_params = ["maximum", "minimum", "coefficients"]

#     def __init__(self, minimum, maximum, coefficients, grid, grid_dmat, validate_args=None):
#         if jnp.ndim(coefficients) < 1:
#             raise ValueError(
#                 "`coefficients` parameter must be at least one-dimensional."
#             )
#         batch_shape, event_shape = coefficients.shape[:-1], coefficients.shape[-1:]

#         self.maximum, self.minimum, self.coefficients = promote_shapes(maximum, minimum, coefficients)
#         self._support = constraints.interval(minimum, maximum)
#         super(BSplineDistribution, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
#         self.grid = grid
#         self.grid_dmat = grid_dmat
#         self.grid_log_pdf = jnp.einsum("i,i...->...", self.coefficients, self.grid_dmat)
#         self.grid_pdf = jnp.exp(self.grid_log_pdf)
#         self.norm = trapezoid(self.grid_pdf, self.grid)
#         self.grid_pdf /= self.norm
#         self.grid_cdf = cumtrapz(self.grid_pdf, self.grid)
#         self.grid_cdf = self.grid_cdf.at[-1].set(1)

#     @constraints.dependent_property(is_discrete=False, event_dim=0)
#     def support(self):
#         return self._support

#     def sample(self, key, sample_shape=()):
#         assert is_prng_key(key)
#         return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

#     def _log_prob_nonorm(self, value):
#         return jnp.interp(value, self.grid, self.grid_log_pdf)

#     @validate_sample
#     def log_prob(self, value):
#         return self._log_prob_nonorm(value) - jnp.log(self.norm)

#     def cdf(self, value):
#         return jnp.interp(value, self.grid, self.grid_cdf)

#     def icdf(self, q):
#         return jnp.interp(q, self.grid_cdf, self.grid)


# class LogXBSplineDistribution(BSplineDistribution):
#     arg_constraints = {
#         "maximum": constraints.real,
#         "minimum": constraints.real,
#         "coefficients": constraints.real_vector,
#         "log_grid": constraints.real_vector,
#         "log_grid_dmat": constraints.real_matrix,
#     }
#     reparametrized_params = ["maximum", "minimum", "coefficients"]
#     def __init__(self, minimum, maximum, coefficients, log_grid, log_grid_dmat, validate_args=None):
#         """
#         grid should be in log(x)
#         """
#         if jnp.ndim(coefficients) < 1:
#             raise ValueError(
#                 "`coefficients` parameter must be at least one-dimensional."
#             )
#         batch_shape, event_shape = coefficients.shape[:-1], coefficients.shape[-1:]

#         self.maximum, self.minimum, self.coefficients = promote_shapes(maximum, minimum, coefficients)
#         self._support = constraints.interval(minimum, maximum)
#         super(BSplineDistribution, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
#         self.log_grid = log_grid
#         self.log_grid_dmat = log_grid_dmat
#         self.log_grid_log_pdf = jnp.einsum("i,i...->...", self.coefficients, self.log_grid_dmat)
#         self.log_grid_pdf = jnp.exp(self.log_grid_log_pdf)
#         self.norm = trapezoid(self.log_grid_pdf, jnp.exp(self.log_grid))
#         self.log_grid_pdf /= self.norm
#         self.log_grid_cdf = cumtrapz(self.log_grid_pdf, jnp.exp(self.log_grid))
#         self.log_grid_cdf = self.log_grid_cdf.at[-1].set(1)

#     def _log_prob_nonorm(self, value):
#         return jnp.interp(jnp.log(value), self.log_grid, self.log_grid_log_pdf)

#     def cdf(self, value):
#         return jnp.interp(jnp.log(value), self.log_grid, self.log_grid_cdf)

#     def icdf(self, q):
#         return jnp.exp(jnp.interp(q, self.log_grid_cdf, self.log_grid))

class BSplineDistribution(Distribution):
    arg_constraints = {
        "maximum": constraints.real,
        "minimum": constraints.real,
        "coefficients": constraints.real_vector,
        "grid": constraints.real_vector,
        "grid_dmat": constraints.real_matrix,
    }
    reparametrized_params = ["maximum", "minimum", "coefficients"]

    def __init__(self, minimum, maximum, coefficients, grid, grid_dmat, validate_args=None):
        if jnp.ndim(coefficients) < 1:
            raise ValueError(
                "`coefficients` parameter must be at least one-dimensional."
            )
        batch_shape, event_shape = coefficients.shape[:-1], coefficients.shape[-1:]

        self.maximum, self.minimum, self.coefficients = promote_shapes(maximum, minimum, coefficients)
        self._support = constraints.interval(minimum, maximum)
        super(BSplineDistribution, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        self.grid = grid
        self.grid_dmat = grid_dmat
        self.grid_pdf = jnp.einsum("i,i...->...", self.coefficients, self.grid_dmat)
        self.grid_log_pdf = jnp.log(self.grid_pdf)
        self.grid_cdf = cumtrapz(self.grid_pdf, self.grid)
        self.grid_cdf = self.grid_cdf.at[-1].set(1)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value):
        return jnp.interp(value, self.grid, self.grid_log_pdf)

    def cdf(self, value):
        return jnp.interp(value, self.grid, self.grid_cdf)

    def icdf(self, q):
        return jnp.interp(q, self.grid_cdf, self.grid)


class LogXBSplineDistribution(Distribution):
    arg_constraints = {
        "maximum": constraints.real,
        "minimum": constraints.real,
        "coefficients": constraints.real_vector,
        "grid": constraints.real_vector,
        "grid_dmat": constraints.real_matrix,
    }
    reparametrized_params = ["maximum", "minimum", "coefficients"]

    def __init__(self, minimum, maximum, coefficients, log_grid, log_grid_dmat, validate_args=None):
        if jnp.ndim(coefficients) < 1:
            raise ValueError(
                "`coefficients` parameter must be at least one-dimensional."
            )
        batch_shape, event_shape = coefficients.shape[:-1], coefficients.shape[-1:]

        self.maximum, self.minimum, self.coefficients = promote_shapes(maximum, minimum, coefficients)
        self._support = constraints.interval(minimum, maximum)
        super(LogXBSplineDistribution, self).__init__(batch_shape=batch_shape, validate_args=validate_args)
        self.log_grid = log_grid
        self.log_grid_dmat = log_grid_dmat
        self.log_grid_pdf = jnp.einsum("i,i...->...", self.coefficients, self.log_grid_dmat)
        self.log_grid_log_pdf = jnp.log(self.log_grid_pdf)
        self.log_grid_cdf = cumtrapz(self.log_grid_pdf, self.log_grid)
        self.log_grid_cdf = self.log_grid_cdf.at[-1].set(1)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(random.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value):
        return jnp.interp(jnp.log(value), self.log_grid, self.log_grid_log_pdf)

    def cdf(self, value):
        return jnp.interp(jnp.log(value), self.log_grid, self.log_grid_cdf)

    def icdf(self, q):
        return jnp.exp(jnp.interp(q, self.log_grid_cdf, self.log_grid))


class _BSplineCoefficient(constraints.Constraint):
    def __init__(self, basis_norms):
        self.basis_norms = basis_norms

    def __call__(self, x):
        prob = (self.basis_norms * x).sum(axis=-1)
        return (prob < 1 + 1e-6) & (prob > 1 - 1e-6)

    def tree_flatten(self):
        return (self.basis_norms,), (("basis_norms",), dict())

    def feasible_like(self, prototype):
        return jnp.full_like(prototype, 1 / prototype.shape[-1])

    def __eq__(self, other):
        if not isinstance(other, _BSplineCoefficient):
            return False
        return jnp.array_equal(self.basis_norms, other.basis_norms)

bspline_constraint = _BSplineCoefficient
@transforms.biject_to.register(bspline_constraint)
def _transform_to_bspline_coefficients(constraint):
    return transforms.ComposeTransform(
        [
            transforms.StickBreakingTransform(),
            transforms.AffineTransform(0, 1/constraint.basis_norms),
        ]
    )


class PSplineCoeficientPrior(Distribution):
    arg_constraints = {"inv_var": constraints.positive}
    reparametrized_params = ["inv_var"]

    def __init__(self, N, inv_var, basis_norms, diff_order=2, validate_args=None):
        (self.inv_var,) = promote_shapes(inv_var)
        self._support = bspline_constraint(basis_norms)
        batch_shape = broadcast_shapes(jnp.shape(inv_var))
        super(PSplineCoeficientPrior, self).__init__(batch_shape=batch_shape, validate_args=validate_args, event_shape=(N,))
        self.diff_order = diff_order
        self.N = N

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value):
        assert value.shape == (self.N,)
        return apply_difference_prior(value, self.inv_var, self.diff_order)