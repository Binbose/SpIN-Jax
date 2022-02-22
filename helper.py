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

def plot_2d_output(model, state, D, n_eigenfunc=0, N=100):

    # generate 2 2d grids for the x & y bounds
    y, x = np.meshgrid(np.linspace(-D, D, N), np.linspace(-D, D, N))
    coordinates = np.stack([x,y], axis=-1).reshape(-1,2)

    z = model.apply(state, (coordinates, D))[:, n_eigenfunc].reshape(N,N)
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('Eigenfunction {}'.format(n_eigenfunc))
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()


if __name__ == '__main__':
    D = 50
    N = 100
    model = EigenNet()
    batch = jnp.ones((N**2, 2))
    variables = model.init(jax.random.PRNGKey(0), (batch, D))
    weight_list = [variables['params'][key]['kernel'] for key in variables['params'].keys()]
    layer_sparsifying_masks = EigenNet().get_all_layer_sparsifying_masks(weight_list, 3)
    weight_list = [w*wm for w, wm in zip(weight_list, layer_sparsifying_masks)]
    output = model.apply(variables, (batch, D))

    plot_2d_output(EigenNet(), variables, D, n_eigenfunc=0, N=N)