import jax
import jax.numpy as jnp                # JAX NumPy

import numpy as np                     # Ordinary NumPy
import matplotlib.pyplot as plt
from backbone import EigenNet
from jax import jvp, grad
from pathlib import Path
from flax.training import checkpoints
from jax import vmap
from jax import jacfwd

def vectorized_diagonal(m):
    return vmap(jnp.diag)(m)

def vectorized_hessian(fn):
    return vmap(jax.hessian(fn))

def get_hessian_diagonals(fn, x):
    vectorized_hessian_result = vectorized_hessian(fn)(x)
    batch, n_eigenfunc, c1, c2 = vectorized_hessian_result.shape[0], vectorized_hessian_result.shape[1], vectorized_hessian_result.shape[2], vectorized_hessian_result.shape[3]
    vectorized_hessian_result = vectorized_hessian_result.reshape(batch*n_eigenfunc, c1, c2)
    return vectorized_diagonal(vectorized_hessian_result).reshape(batch, n_eigenfunc, -1)


def moving_average(running_average, new_data, beta):
    return running_average - beta*(running_average - new_data)



def plot_output(model, weight_dict, D_min, D_max, n_eigenfunc=0, L_inv=None, n_space_dimension=2, N=100, save_dir=None):

    if n_space_dimension == 1:
        x = np.linspace(D_min,D_max, N)[:,None]
        if L_inv is not None:
            z = model.apply(weight_dict, (x, L_inv))[:, n_eigenfunc]
            # print('Min ', jnp.min(z), ' Max ', jnp.max(z))
        else:
            z = model.apply(weight_dict, x)[:, n_eigenfunc]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        plt.plot(x,z)

    elif n_space_dimension == 2:
        # generate 2 2d grids for the x & y bounds
        y, x = np.meshgrid(np.linspace(D_min, D_max, N), np.linspace(D_min, D_max, N))
        coordinates = np.stack([x, y], axis=-1).reshape(-1, 2)

        if L_inv is not None:
            z = model.apply(weight_dict, (coordinates, L_inv))[:, n_eigenfunc].reshape(N, N)
            #print('Min ', jnp.min(z), ' Max ', jnp.max(z))
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


def create_checkpoint(save_dir, model, weight_dict, D_min, D_max, n_space_dimension, opt_state, epoch, sigma_t_bar, j_sigma_t_bar, loss, energies, n_eigenfuncs, L_inv):
    checkpoints.save_checkpoint('{}/checkpoints'.format(save_dir), (weight_dict, opt_state, epoch, sigma_t_bar, j_sigma_t_bar), epoch, keep=2)
    np.save('{}/loss'.format(save_dir), loss), np.save('{}/energies'.format(save_dir), energies)

    for i in range(n_eigenfuncs):
        plot_output(model, weight_dict, D_min, D_max, L_inv=L_inv, n_eigenfunc=i, n_space_dimension=n_space_dimension,
                           N=100, save_dir='{}/eigenfunctions/epoch_{}'.format(save_dir, epoch))

    energies_array = np.array(energies)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i in range(n_eigenfuncs):
        plt.plot(energies_array[:, i], label='Eigenvalue {}'.format(i), )
    plt.legend()
    plt.savefig('{}/energies'.format(save_dir, save_dir))
    plt.close()

    for i in range(n_eigenfuncs):
        plt.plot(energies_array[-500:, i], label='Eigenvalue {}'.format(i))
    plt.legend()
    plt.savefig('{}/energies_newest'.format(save_dir, save_dir))
    plt.close()

    plt.plot(loss)
    plt.savefig('{}/loss'.format(save_dir))
    plt.close()

    plt.plot(loss[-500:])
    plt.savefig('{}/loss_newest'.format(save_dir))
    plt.close()

    np.save('{}/loss'.format(save_dir), loss)
    np.save('{}/energies'.format(save_dir), energies)


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

    plot_2d_output(model, weight_dict, -D,D, n_eigenfunc=2,
                   n_space_dimension=n_space_dimension, N=N)
