from stfn_backbone import STFN_Net
import jax.numpy as jnp
import jax
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.87'

D_min= -50
D_max = 50
batch_size = 2
n_space_dimension = 3

rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)
model = STFN_Net(n_neuron=[128,128,128],n_eigenfuncs=5, D_min=D_min, D_max=D_max)
batch = jnp.ones((batch_size, n_space_dimension))-jnp.array([0,0.2,0.1])
weight_dict = model.init(init_rng, batch)

x = model.apply(weight_dict,jnp.array([[1.11,0.0,0.0]]))

print(x)