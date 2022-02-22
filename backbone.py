import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train weight_dict
from flax.core import FrozenDict

import numpy as np                     # Ordinary NumPy
from typing import Sequence

class EigenNet(nn.Module):
    features: Sequence[int]


    @nn.compact
    def __call__(self, x_in):
        x_in, D = x_in

        x = nn.softplus(nn.Dense(self.features[0])(x_in))
        for feat in self.features[1:-1]:
            x = nn.softplus(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)

        d = jnp.sqrt(2 * D ** 2 - x_in ** 2) - D
        d = jnp.prod(d, axis=-1, keepdims=True)
        x = x * d

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
            ii_in_bound = np.logical_and(ii_is_greater_than_lower_bound, ii_is_smaller_than_upper_bound)
            jj_is_greater_than_lower_bound = jj >= lower_bound_output
            jj_is_smaller_than_upper_bound = jj <= upper_bound_output
            jj_in_bound = np.logical_and(jj_is_greater_than_lower_bound, jj_is_smaller_than_upper_bound)

            if ii_in_any_bound is None:
                ii_in_any_bound = ii_in_bound
                jj_in_any_bound = jj_in_bound
            else:
                ii_in_any_bound = np.logical_or(ii_in_any_bound, ii_in_bound)
                jj_in_any_bound = np.logical_or(jj_in_any_bound, jj_in_bound)

        layer_sparsifing_mask = np.logical_and(ii_in_any_bound, jj_in_any_bound)
        return layer_sparsifing_mask

    @staticmethod
    def get_all_layer_sparsifying_masks(weight_dict, sparsifing_K):
        L = len(weight_dict['params'])
        return [jax.lax.stop_gradient(EigenNet().get_layer_sparsifying_mask(weight_dict['params'][key]['kernel'], sparsifing_K, l, L)) for l, key in enumerate(weight_dict['params'].keys())]

    @staticmethod
    def sparsify_weights(weight_dict, layer_sparsifying_masks):
        weight_dict = weight_dict.unfreeze()
        for key, sparsifying_layer_mask in zip(weight_dict['params'].keys(), layer_sparsifying_masks):
            weight_dict['params'][key]['kernel'] = weight_dict['params'][key]['kernel'] * sparsifying_layer_mask

        weight_dict = FrozenDict(weight_dict)
        return weight_dict



if __name__ == '__main__':
    D = 2
    sparsifying_K = 3
    n_eigenfunc = 9
    model = EigenNet([128,128,128,n_eigenfunc])
    batch = jnp.ones((16, 2))
    weight_dict = model.init(jax.random.PRNGKey(0), (batch, D))
    weight_list = [weight_dict['params'][key]['kernel'] for key in weight_dict['params'].keys()]
    layer_sparsifying_masks = EigenNet().get_all_layer_sparsifying_masks(weight_dict, sparsifying_K)
    weight_list = [w*wm for w, wm in zip(weight_list, layer_sparsifying_masks)]
    output = model.apply(weight_dict, (batch, D))
