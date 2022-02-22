import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
from backbone import EigenNet
import matplotlib.pyplot as plt
from backbone import EigenNet
from train_spin import create_train_state

def plot_2d_output(model, state, N=100):

    # generate 2 2d grids for the x & y bounds
    y, x = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    coordinates = np.stack([x,y], axis=-1).reshape(-1,2)

    z = model.apply(state, coordinates)
    print(z)
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()



rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

learning_rate = 0.1
momentum = 0.9

state = create_train_state(init_rng, learning_rate, momentum)

plot_2d_output(EigenNet(), state)