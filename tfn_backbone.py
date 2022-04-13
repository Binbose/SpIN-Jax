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

import tfn

class TFNEigenNet(nn.Module):
    features: Sequence[int]
    D_min: float  # Dimension of the [-D,D]^2 box where we sample input x_in.
    D_max: float
    radial_channels: Sequence[int] = (32, 32, 32)

    @nn.compact
    def __call__(self, x_in):

        # r : [N, 3]
        # add the nucleus
        r = jnp.concatenate([jnp.zeros((1,3), dtype=jnp.float32), x_in], axis=0)

        # rij : [N, N, 3]
        rij = tfn.difference_matrix(r)
        # print(rij.shape)

        # dij : [N, N]
        dij = tfn.distance_matrix(r)

        N = dij.shape[0]

        radial = dij.reshape(-1, 1) # (N^2, 1)

        for chan in self.radial_channels:
        # radial : [N, N, radial_channel]
            radial = nn.Dense(chan)(radial)
            radial = nn.softplus(radial)
        
        radial = radial.reshape(N, N, -1)
        # embed : [N, layer1_dim, 1]
        ones = jnp.ones((N, 1, 1)) # B, channels, 2L+1
        embed = nn.DenseGeneral(self.features[0], use_bias=False, axis=-2)(ones)
        # embed = tfn.self_interaction_layer(self.features[0], use_bias=False)()

        input_tensor_list = {0: [embed]}

        for layer, layer_dim in enumerate(self.features[1:]):
            input_tensor_list = tfn.convolution()(input_tensor_list, radial, rij)
            input_tensor_list = tfn.concatenation(input_tensor_list)
            input_tensor_list = tfn.self_interaction(layer_dim)(input_tensor_list)
            input_tensor_list = tfn.nonlinearity()(input_tensor_list)

        # x = input_tensor_list[1][0]
        # concatenate l=0 (1 channel), l=1 (3 channels), total is 4 channels
        x = jnp.concatenate([input_tensor_list[0][0], input_tensor_list[1][0]], axis=-1)
        # x: [n_elecs, n_eig, 1+3 channels]
        # print(x.shape)
        
        # dot product over this channel dimension
        ws = self.param('final_weights', nn.initializers.glorot_uniform(), (1+3, 1))
        # ws = jnp.array([[0, 0, 1, 0]], dtype=jnp.float32)
        # print(x.shape, ws.shape)
        x = jnp.einsum('abc,oc->ab', x, ws)
        # x = nn.Dense(1, use_bias=False, kernel_init=nn.initializers.glorot_uniform())(x)
        # x = x.squeeze(axis=-1)
        # x: [n_elecs, n_eig]
        # print(x.shape)

        # only select nucleus
        elec_ws = self.param('elec_weights', nn.initializers.glorot_uniform(), (x.shape[0], 1))
        x = jnp.einsum('eg,oe->g', x, elec_ws)
        # x: [neig]
        # print(x.shape)

        # jastrow = self.param('jastrow', jax.random.uniform)
        # return jnp.exp(-jastrow*jnp.linalg.norm(x_in))*x

        D_avg = (self.D_max + self.D_min) / 2
        lim = self.D_max - D_avg
        d = (jnp.sqrt(2 * lim ** 2 - (x_in.flatten() - D_avg) ** 2) - lim) / lim
        d = jnp.prod(d, axis=-1, keepdims=True)
        # print('d', d.shape) # (1,)
        x = x * d
        
        return x

if __name__ == '__main__':
    import jax
    from jax.config import config
    config.update('jax_platform_name', 'cpu')
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    model = TFNEigenNet([4, 4, 5], -50, 50)
    key = jax.random.PRNGKey(1)
    B = 3000
    D = 3

    x = jax.random.uniform(key, (B, 1, 3), minval=-D, maxval=D)
    # x = coordinates
    # L_inv = jax.random.uniform(key, (2, 2))
    weights = model.init(key, x[0])

    vmodel = jax.vmap(model.apply, in_axes=[None, 0])
    out = vmodel(weights, x)
    out.shape