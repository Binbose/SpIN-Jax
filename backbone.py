import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
# Useful dataclass to keep train weight_dict
from flax.training import train_state
from flax.core import FrozenDict

import numpy as np                     # Ordinary NumPy
from typing import Sequence
from flax.linen import jit
from jax import random, jit, vmap, jacfwd
from jax.nn import sigmoid
from jax.nn import initializers
from jax import dtypes
import matplotlib.pyplot as plt

def constant(value, dtype=jnp.float_):
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * value
  return init

class EigenNet(nn.Module):
    features: Sequence[int]
    D_min: float  # Dimension of the [-D,D]^2 box where we sample input x_in.
    D_max: float
    mask_type = 'quadratic'
    @nn.compact
    def __call__(self, x_in):
        if type(x_in) == tuple:
            x_in, L_inv = x_in
        else:
            L_inv = None
        x = x_in
        x = (x - (self.D_max + self.D_min) / 2) / jnp.max(jnp.array([self.D_max, self.D_min]))
        activation = jax.nn.sigmoid
        '''
        initilization = initializers.variance_scaling
        x = nn.Dense(self.features[0], use_bias=False, kernel_init=initilization(self.features[0], 'fan_out', 'normal'))(x)
        x = activation(x)

        for i,feat in enumerate(self.features[1:-1]):
            x = nn.Dense(feat, use_bias=False, kernel_init=initilization(feat, 'fan_out', 'normal'))(x)
            x = activation(x)
        x = nn.Dense(self.features[-1], use_bias=False, kernel_init=initilization(self.features[-1], 'fan_out', 'normal'))(x)

        '''
        activation = jax.nn.celu
        initilization = initializers.he_normal
        x = nn.Dense(self.features[0], use_bias=False, kernel_init=initilization())(x)
        x = activation(x)

        for i, feat in enumerate(self.features[1:-1]):
            x = nn.Dense(feat, use_bias=False, kernel_init=initilization())(x)
            x = activation(x)
        x = nn.Dense(self.features[-1], use_bias=False, kernel_init=initilization())(x)


        if self.mask_type == 'quadratic':
            # We multiply the output by \prod_i (\sqrt{2D^2-x_i^2}-D) to apply a boundary condition \psi(D_max) = 0 and \psi(D_min) = 0
            # See page 16th for more information
            d = jnp.sqrt(2 * (self.D_max - (self.D_max + self.D_min) / 2) ** 2 - (x_in - (self.D_max + self.D_min) / 2) ** 2) - (self.D_max - (self.D_max + self.D_min) / 2)
            d = jnp.prod(d, axis=-1, keepdims=True)
            x = x * d
        elif self.mask_type == 'exp':
            # Mask with gaussian instead to satisfy boundary condition \psi(x) -> 0 for x -> \infty
            # Standard deviation of gaussian is learnable
            mean = (self.D_max + self.D_min) / 2
            sigma = jnp.max(jnp.array([self.D_max, self.D_min])) / 4

            #embedding = jnp.abs(nn.Embed(1, self.features[-1], embedding_init=constant(sigma))(jnp.eye(1, dtype='int32')))
            #sigma = (embedding * jnp.eye(k))[0]
            #print(embedding)

            normalization = 1 / (jnp.sqrt(2*jnp.pi) * sigma)
            d = normalization * jnp.exp(-0.5 * jnp.linalg.norm(x_in-mean, axis=-1, keepdims=True)**2 / sigma**2)
            x = x * d

        if L_inv is not None:
            x = jnp.einsum('ij, bj -> bi', L_inv, x)

        return x

    @staticmethod
    def get_layer_sparsifying_mask(W, sparsifing_K, l, L):
        l += 1
        m = W.shape[0]
        n = W.shape[1]

        x = np.linspace(0, m-1, m)
        y = np.linspace(0, n-1, n)
        ii, jj = np.meshgrid(x, y, indexing='ij')

        layer_sparsifing_mask = None
        beta_0 = (l - 1)/L
        beta_1 = (L - l + 1) / L
        for k in range(1, sparsifing_K+1):
            alpha = (k-1)/(sparsifing_K-1) * beta_0

            lower_bound_input = alpha * m
            upper_bound_input = (alpha + beta_1) * m
            lower_bound_output = alpha * n
            upper_bound_output = (alpha + beta_1) * n

            ii_is_greater_than_lower_bound = ii >= lower_bound_input
            ii_is_smaller_than_upper_bound = ii <= upper_bound_input
            ii_in_bound = np.logical_and(ii_is_greater_than_lower_bound, ii_is_smaller_than_upper_bound)

            jj_is_greater_than_lower_bound = jj >= lower_bound_output
            jj_is_smaller_than_upper_bound = jj <= upper_bound_output
            jj_in_bound = np.logical_and(jj_is_greater_than_lower_bound, jj_is_smaller_than_upper_bound)

            if layer_sparsifing_mask is None:
                layer_sparsifing_mask = np.logical_and(ii_in_bound, jj_in_bound)
            else:
                layer_sparsifing_mask_ = np.logical_and(ii_in_bound, jj_in_bound)
                layer_sparsifing_mask = np.logical_or(layer_sparsifing_mask, layer_sparsifing_mask_)


        return layer_sparsifing_mask

    @staticmethod
    def get_all_layer_sparsifying_masks(weight_dict, sparsifing_K: int):
        L = len(weight_dict['params'])
        return [jax.lax.stop_gradient(EigenNet.get_layer_sparsifying_mask(weight_dict['params'][key]['kernel'], sparsifing_K, l, L)) for l, key in enumerate(weight_dict['params'].keys())]

    @staticmethod
    def sparsify_weights(weight_dict, layer_sparsifying_masks):
        weight_dict = weight_dict.unfreeze()
        for key, sparsifying_layer_mask in zip(weight_dict['params'].keys(), layer_sparsifying_masks):
            weight_dict['params'][key]['kernel'] = weight_dict['params'][key]['kernel'] * sparsifying_layer_mask

        weight_dict = FrozenDict(weight_dict)
        return weight_dict


if __name__ == '__main__':
    D = 50
    sparsifying_K = 3
    n_eigenfuncs = 9
    model = EigenNet(features=[128, 128, 128, n_eigenfuncs], D=D)
    batch = jnp.ones((16, 2))
    weight_dict = model.init(jax.random.PRNGKey(0), batch)
    weight_list = [weight_dict['params'][key]['kernel']
                   for key in weight_dict['params'].keys()]
    layer_sparsifying_masks = EigenNet.get_all_layer_sparsifying_masks(
        weight_dict, sparsifying_K)
    weight_dict = EigenNet.sparsify_weights(
        weight_dict, layer_sparsifying_masks)
    output = model.apply(weight_dict, batch)
