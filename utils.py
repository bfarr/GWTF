from jax import jit, vmap
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import equinox as eqx
from gwinferno.preprocess.data_collection import dl_2_prior_on_z


def cumtrapz(y, x):
    difs = jnp.diff(x)
    idxs = jnp.array([i for i in range(1,len(y))])
    res = jnp.cumsum(vmap(lambda i,d: d * (y[i] + y[i+1]) / 2.0)(idxs,difs))
    return jnp.concatenate([jnp.array([0]), res])


def logit_transform(x, low, high):
    return jnp.log((x-low)/(high-x))


def logit_transform_jacobian(x, low, high):
    return (high-low) / (x-low) / (high-x)


def inv_logit_transform(x, low, high):
    return low + (high-low) / (1 + jnp.exp(-x))


def chirp_mass_from_m1q(m1, q):
    return m1 * q**(0.6) / (1+q)**(0.2)


def chirp_mass_from_m1q_jacobian(q):
    return q**(0.6) / (1+q)**(0.2)


def m1_from_chirp_massq(mc, q):
    return mc * (1.+q)**0.2 / q**0.6


def remove_bounds(post, bounds):
    for p, b in bounds.items():
        if None not in b:
            x = post.pop(p)
            post[f"{p}_logit"] = logit_transform(x, b[0], b[1])
    return post


@jit
def source_to_nf(m1, q, a1, a2, ct1, ct2, z):
    mc = chirp_mass_from_m1q(m1, q)
    q_logit = logit_transform(q, 0, 1)
    a1_logit = logit_transform(a1, 0, 1)
    a2_logit = logit_transform(a2, 0, 1)
    ct1_logit = logit_transform(ct1, -1, 1)
    ct2_logit = logit_transform(ct2, -1, 1)
    jacs = jnp.log(chirp_mass_from_m1q_jacobian(q))
    jacs += jnp.log(logit_transform_jacobian(q, 0, 1))
    jacs += jnp.log(logit_transform_jacobian(a1, 0, 1))
    jacs += jnp.log(logit_transform_jacobian(a2, 0, 1))
    jacs += jnp.log(logit_transform_jacobian(ct1, -1, 1))
    jacs += jnp.log(logit_transform_jacobian(ct2, -1, 1))
    x = jnp.array([mc, q_logit, a1_logit, a2_logit, ct1_logit, ct2_logit, z]).T.reshape((*m1.shape, 7))
    return x, jacs


@eqx.filter_jit
def nf_to_source(x_logit):
    mc, q_logit, a1_logit, a2_logit, ct1_logit, ct2_logit, z = x_logit.T
    q = inv_logit_transform(q_logit, 0, 1)
    m1 = m1_from_chirp_massq(mc, q)
    a1 = inv_logit_transform(a1_logit, 0, 1)
    a2 = inv_logit_transform(a2_logit, 0, 1)
    ct1 = inv_logit_transform(ct1_logit, -1, 1)
    ct2 = inv_logit_transform(ct2_logit,  -1, 1)
    return m1, q, a1, a2, ct1, ct2, z


@eqx.filter_vmap
def compute_cat_mean_cov(data):
    return jnp.mean(data, axis=0), jnp.cov(data.T)


_zs = jnp.linspace(1e-8, 2.3, 1000)
_pzs = dl_2_prior_on_z(_zs)
_pzs /= trapezoid(_pzs, _zs)
def dl_2_prior_on_z(z):
    return jnp.interp(z, _zs, _pzs)