import time
import jax.numpy as jnp
from jax import random, jit, vmap, jacfwd
from jax.nn import sigmoid, softplus
import jax
from flax import linen as nn
import numpy as np
from typing import Sequence
import helper
from physics import hamiltonian_operator

def MLP(layers):
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
            outputs = jnp.dot(inputs, W) #+ b
            inputs = sigmoid(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) #+ b
        return outputs
    return init, apply


class EigenNet(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x_in):
        x = nn.Dense(self.features[0], use_bias=False)(x_in)
        x = sigmoid(x)

        for feat in self.features[1:-1]:
            x = nn.Dense(feat, use_bias=False)(x)
            x = sigmoid(x)
        x = nn.Dense(self.features[-1], use_bias=False)(x)

        return x

def get_network_as_function_of_input(model_apply, params):
    return lambda batch: model_apply(params, batch)

r=5
rng = jax.random.PRNGKey(r)
rng, init_rng = jax.random.split(rng)
D = np.pi
#inputs = jax.random.uniform(rng, minval=-D, maxval=D, shape=(128, 1))
inputs = np.load('./batch.npy')

test_case = 'derivatives'

layers = [1, 64, 64, 64, 32, 4]
net_init, net_apply = MLP(layers)
params = net_init(random.PRNGKey(r))
params = weight_list = np.load('./weights.npy', allow_pickle=True)

model = EigenNet(features=[64, 64, 64, 32, 4])
weight_dict = model.init(rng, inputs)
weight_dict = weight_dict.unfreeze()
weight_list = params
for i, key in enumerate(weight_dict['params'].keys()):
    w, b = weight_list[i]
    weight_dict['params'][key]['kernel'] = w
flax_apply = lambda weight_dict, inputs: model.apply(weight_dict, inputs)

if test_case == 'times':
    outputs = net_apply(params, inputs)

    # jit and pre-compile (we don't want to compare compile times)
    net_apply_jitted = jax.jit(net_apply)
    outputs = net_apply_jitted(params, inputs)

    t1 = time.time()
    outputs = net_apply(params, inputs)
    #print('TIME JAX ', time.time()-t1)

    t1 = time.time()
    outputs = net_apply_jitted(params, inputs).block_until_ready()
    #print(outputs.sum())
    #print('TIME JAX JITTED', time.time()-t1)



    #############################################################################



    # jit and pre-compile
    flax_apply_jitted = jax.jit(lambda weight_dict, inputs: model.apply(weight_dict, inputs))
    _ = flax_apply_jitted(weight_dict, inputs)

    _ = model.apply(weight_dict, inputs)
    t1 = time.time()
    _ = model.apply(weight_dict, inputs)
    #print('TIME FLAX ', time.time()-t1)

    t1 = time.time()
    outputs = flax_apply_jitted(weight_dict, inputs).block_until_ready()
    #print(outputs.sum())
    #print('TIME FLAX JITTED', time.time()-t1)


elif test_case == 'derivatives':
    u_of_x = get_network_as_function_of_input(net_apply, params)
    outputs = u_of_x(inputs)
    print('OUT 1 ', outputs[:3])
    print()
    h = hamiltonian_operator(net_apply, u_of_x, inputs, params, fn_x=outputs, system='laplace', nummerical_diff=True,
                             eps=0.01)
    print('\n\n')


    u_of_x = get_network_as_function_of_input(flax_apply, weight_dict)
    outputs = u_of_x(inputs)
    print('OUT 2 ', outputs[:3])
    print()
    h = hamiltonian_operator(flax_apply, u_of_x, inputs, weight_dict, fn_x=outputs, system='laplace',
                             nummerical_diff=True, eps=0.01)
