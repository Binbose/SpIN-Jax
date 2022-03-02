import jax
import jax.numpy as jnp                # JAX NumPy

import numpy as np                     # Ordinary NumPy
import matplotlib.pyplot as plt
from backbone import EigenNet
from jax import jvp, grad
from pathlib import Path

def get_hessian_diagonals(fn, x):
    return jnp.diag(jax.hessian(fn)(x))


def hvp(f, x, v):
    return jvp(grad(f), (x,), (v,))[1]


def get_hessian_diagonals_2(fn, x):
    return hvp(fn, x, jnp.ones_like(x))


def moving_average(running_average, new_data, beta):
    return running_average - beta*(running_average - new_data)


def plot_2d_output(model, weight_dict, D, n_eigenfunc=0, L_inv=None, n_space_dimension=2, N=100, save_dir=None):

    # generate 2 2d grids for the x & y bounds
    y, x = np.meshgrid(np.linspace(-D, D, N), np.linspace(-D, D, N))
    coordinates = np.stack([x, y], axis=-1).reshape(-1, 2)

    if L_inv is not None:
        z = model.apply(weight_dict, (coordinates, L_inv))[:, n_eigenfunc].reshape(N, N)
        print(z)
    else:
        z = model.apply(weight_dict, coordinates)[:, n_eigenfunc].reshape(N, N)
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('Eigenfunction {}'.format(n_eigenfunc))
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    if save_dir is None:
        plt.show()
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/eigenfunc_{}'.format(save_dir, n_eigenfunc))
        plt.close()


if __name__ == '__main__':
    D = 50
    N = 100
    sparsifying_K = 3
    n_eigenfunc = 9
    n_space_dimension = 2
    model = EigenNet(features=[128, 128, 128, n_eigenfunc], D=D)
    batch = jnp.ones((N**2, n_space_dimension))
    weight_dict = model.init(jax.random.PRNGKey(0), batch)
    weight_list = [weight_dict['params'][key]['kernel']
                   for key in weight_dict['params'].keys()]
    layer_sparsifying_masks = EigenNet.get_all_layer_sparsifying_masks(
        weight_dict, sparsifying_K)
    weight_dict = EigenNet.sparsify_weights(
        weight_dict, layer_sparsifying_masks)
    output = model.apply(weight_dict, batch)

    plot_2d_output(model, weight_dict, D, n_eigenfunc=2,
                   n_space_dimension=n_space_dimension, N=N)
