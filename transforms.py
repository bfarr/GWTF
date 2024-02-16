from numpyro.distributions import (
    transforms,
    constraints,
)
from jax import jit


class FlowMCTransform(transforms.Transform):
    """
    A wrapper of flowMCs rqSpine flow. Since this is
    intended for enabling sampling in the latent space,
    we'll swap forward/inverse definitions from flowMC's
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, ensembled_flow):
        self.flow = ensembled_flow

    def __call__(self, x):
        """
        :param numpy.ndarray x: the input into the transform
        """
        return self.call_with_intermediates(x)[0]

    @jit
    def call_with_intermediates(self, x):
        """
        :param numpy.ndarray x: the input into the transform

        from latent space to data space
        """
        y, log_det = self.flow.inverse_ensemble(x)
        return y, log_det

    @jit
    def _inverse(self, y):
        """
        :param numpy.ndarray y: the output of the transform to be inverted

        from data to latent space
        """
        x, log_det = self.flow.forward_ensemble(y)
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """
        Calculates the elementwise determinant of the log jacobian.

        :param numpy.ndarray x: the input to the transform
        :param numpy.ndarray y: the output of the transform
        """
        if intermediates is None:
            log_det = self.flow.inverse_ensemble(x)[1]
            return log_det
        else:
            log_det = intermediates
            return log_det

    def tree_flatten(self):
        return (), ((), {"flow": self.flow},)

    def __eq__(self, other):
        return (
            isinstance(other, FlowMCTransform)
            and self.flow is other.flow
        )

