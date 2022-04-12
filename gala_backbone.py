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

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, Dense
from geometric_algebra_attention.jax import VectorAttention
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.77'
# rank = 3
# n_dim = 32
# dilation = 2.0
# dilation_dim = int(np.round(n_dim*dilation))
# merge_fun = 'concat'
# join_fun = 'concat'
# n_blocks = 1
# block_nonlinearity = True
# residual = True
# invar_mode = 'full'
# neig = 4

def make_gala_net(
    neig,
    D_min,
    D_max,
    rank=3,
    n_dim=32,
    dilation=2.0,
    merge_fun='concat',
    join_fun='concat',
    n_blocks=2,
    block_nonlinearity=True,
    residual=True,
    invar_mode='full',
):
    dilation_dim = int(np.round(n_dim*dilation))
    
    def make_layernorm():
        def init(rng, input_shape):
            return input_shape, ()

        def eval_(params, x, rng=None):
            return jax.nn.normalize(x)

        return init, eval_

    def make_swish():
        def init(rng, input_shape):
            return input_shape, ()

        def eval_(params, x, rng=None):
            return jax.nn.swish(x)

        return init, eval_

    score = serial(
        Dense(dilation_dim),
        make_swish(),
        Dense(1)
        )

    value = serial(
        Dense(dilation_dim),
        make_layernorm(),
        make_swish(),
        Dense(n_dim)
        )

    def make_attention(reduce=False):
        attention = VectorAttention(
            score, value, reduce=reduce, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode).stax_functions
        return attention

    def init(rng, input_shape):
        query, coord = input_shape
        r_shape, v_shape = query
        coord_r_shape, coord_v_shape = coord

        def rngs_(rng):
            while True:
                (next_rng, rng) = jax.random.split(rng)
                yield next_rng
        rngs = rngs_(rng)

        def param(layer, sh):
            (last_shape, p) = layer[0](next(rngs), sh)
            params.append(p)
            return last_shape

        params = []

        last_shape = param(vscale, v_shape)
        coord_last_shape = param(coord_vscale, coord_v_shape)
        for i, att in enumerate(attentions):
            last_shape = param(att, (r_shape, last_shape, coord_r_shape, coord_last_shape))
            if block_nonlinearity:
                last_shape = param(block_nonlins[i], last_shape)
        last_shape = param(final_attention, (r_shape, last_shape, coord_r_shape, coord_last_shape))
        last_shape = param(final_mlp, last_shape)
        # last_shape = last_shape[:1] + last_shape[2:]
        return last_shape, params

    def init_from_pos(rng, r_shape):
        if len(r_shape) == 2:
            B, n_directions = r_shape
            n_elecs = n_directions // 3
            my_r_shape = (B, n_elecs, 3)
            n_types = n_elecs+3
            x_shape = [(my_r_shape, (B, n_elecs, n_types)), ((B, 3, 3), (B, 3, n_types))]
            return init(rng, x_shape)
        elif len(r_shape) == 1:
            n_directions = r_shape[0]
            n_elecs = n_directions // 3
            my_r_shape = (n_elecs, 3)
            n_types = n_elecs+3
            x_shape = [(my_r_shape, (n_elecs, n_types)), ((3, 3), (3, n_types))]
            return init(rng, x_shape)

    def eval_(params, x, rng=None):
        pstack = list(reversed(params))

        def run(layer, x):
            return layer[1](pstack.pop(), x)

        query, coord = x
        r, v = query
        coord_r, coord_v = coord

        last = run(vscale, v)
        coord_last = run(coord_vscale, coord_v)
        for i, att in enumerate(attentions):
            residual_in = last
            last = run(att, [(r, last), (coord_r, coord_last)])
            if block_nonlinearity:
                last = run(block_nonlins[i], last)
            if residual:
                last = residual_in + last
        last = run(final_attention, [(r, last), (coord_r, coord_last)])
        last = run(final_mlp, last)

        # n_elecs = r.shape[1]
        # if n_elecs == 1:
        #     last = last[:,0]
        # last = 0
        # for i in range(n_elecs):
        #     for j in range(i+1, n_elecs):
        #         last += jax.nn.swish(last[:,i,:] - last[:,j,:])

        # We multiply the output by \prod_i (\sqrt{2D^2-x_i^2}-D) to apply a boundary condition \psi(D_max) = 0 and \psi(D_min) = 0
        # See page 16th for more information
        D_avg = (D_max + D_min) / 2
        lim = D_max - D_avg
        d = (jnp.sqrt(2 * lim ** 2 - (last - D_avg) ** 2) - lim) / lim
        d = jnp.prod(d, axis=-1, keepdims=True) 
        last = last * d
        return last
    
    def eval_from_pos(params, rs, rng=None):
        if type(rs) == tuple:
            rs, L_inv = rs
        else:
            L_inv = None
        if len(rs.shape) == 2:
            B = rs.shape[0]
            my_rs = rs.reshape((B, -1, 3))
            n_elecs = my_rs.shape[1]

            per_example = jnp.concatenate((jnp.eye(n_elecs), jnp.zeros((n_elecs, 3))), axis=1)
            vs = jnp.repeat(per_example[jnp.newaxis,...], B, axis=0)
            coordinate_rs = jnp.repeat(jnp.eye(3)[jnp.newaxis,...], B, axis=0)
            coordinate_vs = jnp.repeat((jnp.concatenate((jnp.zeros((3, n_elecs)), jnp.eye(3)), axis=1))[jnp.newaxis,...], B, axis=0)
            
            x = [(my_rs, vs), (coordinate_rs, coordinate_vs)]

        elif len(rs.shape) == 1:
            my_rs = rs.reshape((-1, 3))
            n_elecs = my_rs.shape[0]

            vs = jnp.concatenate((jnp.eye(n_elecs), jnp.zeros((n_elecs, 3))), axis=1)
            coordinate_rs = jnp.eye(3)
            coordinate_vs = jnp.concatenate((jnp.zeros((3, n_elecs)), jnp.eye(3)), axis=1)
            
            x = [(my_rs, vs), (coordinate_rs, coordinate_vs)]

        out = eval_(params, x, rng)

        if L_inv is not None:
            out = jnp.einsum('ij, bj -> bi', L_inv, out)
        return out
        # elif len(rs.shape) == 2:
        #     vs = jnp.array([[1, 0, 0, 0]], dtype=jnp.float32)
        #     coordinate_rs = jnp.eye(3)
        #     coordinate_vs = jnp.concatenate((jnp.zeros((3, 1)), jnp.eye(3)), axis=1)
            
        #     x = [(rs, vs), (coordinate_rs, coordinate_vs)]
        # return eval_(params, x, rng)
        


    vscale = Dense(n_dim)
    coord_vscale = Dense(n_dim)
    attentions = [make_attention() for _ in range(n_blocks)]
    block_nonlins = []
    if block_nonlinearity:
        block_nonlins = n_blocks*[value]
    final_attention = make_attention(True)
    final_mlp = serial(
        Dense(dilation_dim), make_swish(), Dense(neig))

    return init_from_pos, eval_from_pos

def constant(value, dtype=jnp.float_):
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * value
  return init

if __name__ == '__main__':
    from jax import config
    config.update('jax_platform_name', 'cpu')
    # init_fun, eval_fun = make_gala_net(4, -50, 50)
    # rs = jax.random.normal(jax.random.PRNGKey(0), (10, 1, 3))
    # vs = jnp.array([[[1, 0, 0, 0]]]*10, dtype=jnp.float32)
    # coordinate_rs = jnp.repeat(jnp.eye(3)[jnp.newaxis,...], 10, axis=0)
    # coordinate_vs = jnp.repeat((jnp.concatenate((jnp.zeros((3, 1)), jnp.eye(3)), axis=1))[jnp.newaxis,...], 10, axis=0)
    # x = [(rs, vs), (coordinate_rs, coordinate_vs)]
    # x_shape = jax.tree_multimap(lambda a: a.shape, x)
    # params = init_fun(jax.random.PRNGKey(1), x_shape)[1]
    # print(eval_fun(params, x))
    init_fun, eval_fun = make_gala_net(4, -50, 50)
    B = 10
    n_elecs = 1
    rs = jax.random.normal(jax.random.PRNGKey(0), (B, int(n_elecs*3)))
    # rs = jax.random.normal(jax.random.PRNGKey(0), (B, 1, 3))
    last_shape, params = init_fun(jax.random.PRNGKey(1), rs.shape)
    print(jax.tree_map(lambda x: x.shape, params))
    print(eval_fun(params, rs))
    print()