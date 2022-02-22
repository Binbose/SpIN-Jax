import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers

class EigenNet(nn.Module):

    def setup(self):

        self.dense1 = nn.Dense(128, use_bias=False)
        self.dense2 = nn.Dense(128, use_bias=False)
        self.dense3 = nn.Dense(128, use_bias=False)
        self.dense4 = nn.Dense(9, use_bias=False)



    def __call__(self, x_in):
        x_in, D = x_in

        x = self.dense1(x_in)
        x = nn.softplus(x)
        x = self.dense2(x)
        x = nn.softplus(x)
        x = self.dense3(x)
        x = nn.softplus(x)
        x = self.dense4(x)

        d = jnp.sqrt(2*D**2 - x_in**2) - D
        d = jnp.prod(d, axis=-1, keepdims=True)
        x = x*d

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
    def get_all_layer_sparsifying_masks(weight_list, sparsifing_K):
        L = len(weight_list)
        return [jax.lax.stop_gradient(EigenNet().get_layer_sparsifying_mask(w, sparsifing_K, l, L)) for l, w in enumerate(weight_list)]







if __name__ == '__main__':

    D = 2
    model = EigenNet()
    batch = jnp.ones((16, 2))
    variables = model.init(jax.random.PRNGKey(0), (batch, D))
    weight_list = [variables['params'][key]['kernel'] for key in variables['params'].keys()]
    layer_sparsifying_masks = EigenNet().get_all_layer_sparsifying_masks(weight_list, 3)
    weight_list = [w*wm for w, wm in zip(weight_list, layer_sparsifying_masks)]
    output = model.apply(variables, (batch, D))
