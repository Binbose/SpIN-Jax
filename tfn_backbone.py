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

def constant(value, dtype=jnp.float_):
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * value
  return init

class TFNEigenNet(nn.Module):
    features: Sequence[int]
    D_min: float  # Dimension of the [-D,D]^2 box where we sample input x_in.
    D_max: float

    @nn.compact
    def __call__(self, x_in):
        if type(x_in) == tuple:
            x_in, L_inv = x_in
        else:
            L_inv = None
        
        # radial basis functions
        rbf_low = 0.0
        rbf_high = 3.5
        rbf_count = 4
        rbf_spacing = (rbf_high - rbf_low) / rbf_count
        centers = jnp.linspace(rbf_low, rbf_high, rbf_count)

        # r : [N, 3]
        r = x_in

        # rij : [N, N, 3]
        rij = tfn.difference_matrix(x_in)

        # dij : [N, N]
        dij = tfn.distance_matrix(r)

        # rbf : [N, N, rbf_count]
        gamma = 1. / rbf_spacing
        rbf = jnp.exp(-gamma * jnp.square(jnp.expand_dims(dij, axis=-1) - centers))

        # embed : [N, layer1_dim, 1]
        embed = tfn.self_interaction_layer(self.features[0], use_bias=False)(jnp.ones((r.shape[-2], 1, 1)))

        input_tensor_list = {0: [embed]}

        for layer, layer_dim in enumerate(self.features[1:]):
            # with tf.variable_scope(None, 'layer' + str(layer), values=[input_tensor_list]):
            input_tensor_list = tfn.convolution()(input_tensor_list, rbf, rij)
            input_tensor_list = tfn.concatenation(input_tensor_list)
            input_tensor_list = tfn.self_interaction(layer_dim)(input_tensor_list)
            input_tensor_list = tfn.nonlinearity()(input_tensor_list)

        print(jax.tree_map(lambda x: x.shape, input_tensor_list))
        x = input_tensor_list[1][0]
        # x = jnp.mean(jnp.squeeze(tfn_scalars), axis=-1)
        print(x.shape)

        # D_avg = (self.D_max + self.D_min) / 2
        # lim = self.D_max - D_avg
        # d = (jnp.sqrt(2 * lim ** 2 - (x_in - D_avg) ** 2) - lim) / lim
        # d = jnp.prod(d, axis=-1, keepdims=True) 
        # x = x * d

        if L_inv is not None:
            x = jnp.einsum('ij, bj -> bi', L_inv, x)
        
        return x
