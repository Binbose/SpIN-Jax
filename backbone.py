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


def JaxEigenNeet(layers):
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W = random.normal(k1, (d_in, d_out))
            b = random.normal(k2, (d_out,))
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = sigmoid(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs
    return init, apply



class EigenNet(nn.Module):
    features: Sequence[int]
    D: int  # Dimension of the [-D,D]^2 box where we sample input x_in.

    @nn.compact
    def __call__(self, x_in):
        if type(x_in) == tuple:
            x_in, L_inv = x_in
        else:
            L_inv = None
        x = nn.Dense(self.features[0], use_bias=False, kernel_init=jax.nn.initializers.lecun_normal())(x_in)

        # softplus of x is log(1+exp(x))
        x = nn.sigmoid(x)

        for feat in self.features[1:-1]:
            x = nn.Dense(feat, use_bias=False, kernel_init=jax.nn.initializers.lecun_normal())(x)
            x = nn.sigmoid(x)
        x = nn.Dense(self.features[-1], use_bias=False, kernel_init=jax.nn.initializers.lecun_normal())(x)

        # We multiply the output by \prod_i (\sqrt{2D^2-x_i^2}-D) to apply a boundary condition
        # See page 16th for more information
        d = jnp.sqrt(2 * self.D ** 2 - x_in ** 2) - self.D
        d = jnp.prod(d, axis=-1, keepdims=True)
        x = x * d

        if L_inv is not None:
            x = jnp.einsum('ij, bj -> bi', L_inv, x)
        return x

    @staticmethod
    def get_layer_sparsifying_mask(W, sparsifing_K, l, L):
        m = W.shape[0]
        n = W.shape[1]

        x = np.linspace(0, m-1, m)
        y = np.linspace(0, n-1, n)
        ii, jj = np.meshgrid(x, y, indexing='ij')

        ii_in_any_bound = None
        jj_in_any_bound = None
        beta = (L - l + 1) / L
        for k in range(sparsifing_K):
            alpha = k/(sparsifing_K - 1) * (l - 1)/L

            lower_bound_input = alpha * m
            upper_bound_input = (alpha + beta) * m
            lower_bound_output = alpha * n
            upper_bound_output = (alpha + beta) * n

            ii_is_greater_than_lower_bound = ii >= lower_bound_input
            ii_is_smaller_than_upper_bound = ii <= upper_bound_input
            ii_in_bound = np.logical_and(
                ii_is_greater_than_lower_bound, ii_is_smaller_than_upper_bound)
            jj_is_greater_than_lower_bound = jj >= lower_bound_output
            jj_is_smaller_than_upper_bound = jj <= upper_bound_output
            jj_in_bound = np.logical_and(
                jj_is_greater_than_lower_bound, jj_is_smaller_than_upper_bound)

            if ii_in_any_bound is None:
                ii_in_any_bound = ii_in_bound
                jj_in_any_bound = jj_in_bound
            else:
                ii_in_any_bound = np.logical_or(ii_in_any_bound, ii_in_bound)
                jj_in_any_bound = np.logical_or(jj_in_any_bound, jj_in_bound)

        layer_sparsifing_mask = np.logical_and(
            ii_in_any_bound, jj_in_any_bound)
        return layer_sparsifing_mask

    @staticmethod
    def get_all_layer_sparsifying_masks(weight_dict, sparsifing_K: int):
        L = len(weight_dict['params'])
        return [jax.lax.stop_gradient(EigenNet.get_layer_sparsifying_mask(weight_dict['params'][key]['kernel'], sparsifing_K, l, L)) for l, key in enumerate(weight_dict['params'].keys())]

    @staticmethod
    def sparsify_weights(weight_dict, layer_sparsifying_masks):
        weight_dict = weight_dict.unfreeze()
        for key, sparsifying_layer_mask in zip(weight_dict['params'].keys(), layer_sparsifying_masks):
            weight_dict['params'][key]['kernel'] = weight_dict['params'][key]['kernel'] * \
                sparsifying_layer_mask

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
