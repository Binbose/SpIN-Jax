import time
import jax.numpy as jnp
from jax import random, jit, vmap, jacfwd
from jax.nn import sigmoid, softplus
import jax
from flax import linen as nn
import numpy as np
from typing import Sequence

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
            outputs = jnp.dot(inputs, W) + b
            inputs = sigmoid(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
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

r=5
rng = jax.random.PRNGKey(r)
rng, init_rng = jax.random.split(rng)
D = np.pi

layers = [1, 64, 64, 64, 32, 4]
net_init, net_apply = MLP(layers)
params = net_init(random.PRNGKey(r))

inputs = jax.random.uniform(rng, minval=-D, maxval=D, shape=(128, 1))
outputs = net_apply(params, inputs)

# jit and pre-compile (we don't want to compare compile times)
net_apply_jitted = jax.jit(net_apply)
outputs = net_apply_jitted(params, inputs)

t1 = time.time()
outputs = net_apply(params, inputs)
print('TIME JAX ', time.time()-t1)

t1 = time.time()
outputs = net_apply_jitted(params, inputs).block_until_ready()
print(outputs.sum())
print('TIME JAX JITTED', time.time()-t1)

#############################################################################

model = EigenNet(features=[64, 64, 64, 32, 4])
params = model.init(rng, inputs)

# jit and pre-compile
flax_apply_jitted = jax.jit(lambda params, inputs: model.apply(params, inputs))
_ = flax_apply_jitted(params, inputs)

_ = model.apply(params, inputs)
t1 = time.time()
_ = model.apply(params, inputs)
print('TIME FLAX ', time.time()-t1)

t1 = time.time()
outputs = flax_apply_jitted(params, inputs).block_until_ready()
print(outputs.sum())
print('TIME FLAX JITTED', time.time()-t1)