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


def plot_output(model, weight_dict, D_min, D_max, fig, ax, n_eigenfunc=0, L_inv=None, n_space_dimension=2, N=100):

    if n_space_dimension == 1:
        x = np.linspace(D_min,D_max, N)[:,None]
        if L_inv is not None:
            z = model.apply(weight_dict, (x, L_inv))[:, n_eigenfunc]
            # print('Min ', jnp.min(z), ' Max ', jnp.max(z))
        else:
            z = model.apply(weight_dict, x)[:, n_eigenfunc]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        ax.plot(x,z)

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


        c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('Eigenfunction {}'.format(n_eigenfunc))
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])


def create_plots(n_space_dimension, neig):
    energies_fig, energies_ax = plt.subplots(1, 1)
    if n_space_dimension == 1:
        fig, ax = plt.subplots(1, 1)
        return fig, ax, energies_fig, energies_ax
    elif n_space_dimension == 2:
        nfig = max(2, int(np.ceil(np.sqrt(neig))))
        psi_fig, psi_ax = plt.subplots(nfig, nfig, figsize=(10, 10))
        for ax in psi_ax.flatten():
            ax.set_aspect('equal', adjustable='box')
        return psi_fig, psi_ax, energies_fig, energies_ax

def uniform_sliding_average(data, window):
    ret = np.cumsum(data, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

def uniform_sliding_stdev(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return np.std(rolling, 1)

def create_checkpoint(save_dir, model, weight_dict, D_min, D_max, n_space_dimension, opt_state, epoch, sigma_t_bar, j_sigma_t_bar, loss, energies, n_eigenfuncs, L_inv, window, psi_fig, psi_ax, energies_fig, energies_ax):
    checkpoints.save_checkpoint('{}/checkpoints'.format(save_dir), (weight_dict, opt_state, epoch, sigma_t_bar, j_sigma_t_bar), epoch, keep=2)
    np.save('{}/loss'.format(save_dir), loss), np.save('{}/energies'.format(save_dir), energies)

    if n_space_dimension == 1:
        psi_ax.cla()
    for i in range(n_eigenfuncs):
        if n_space_dimension == 2:
            ax = psi_ax.flatten()[i]
        else:
            ax = psi_ax
        plot_output(model, weight_dict, D_min, D_max, psi_fig, ax, L_inv=L_inv, n_eigenfunc=i, n_space_dimension=n_space_dimension, N=100)
    eigenfunc_dir = f'{save_dir}/eigenfunctions'
    Path(eigenfunc_dir).mkdir(parents=True, exist_ok=True)
    psi_fig.savefig(f'{eigenfunc_dir}/epoch_{epoch}.png')

    energies_array = np.array(energies)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    energies_ax.cla()
    window = 100
    color = plt.cm.tab10(np.arange(n_eigenfuncs))
    for i, c in zip(range(n_eigenfuncs), color):
        x = np.arange(window//2 - 1, len(energies_array[:, i])-(window//2))
        av = uniform_sliding_average(energies_array[:, i], window)
        stdev = uniform_sliding_stdev(energies_array[:, i], window)
        energies_ax.plot(x, av, c=c, label='Eigenvalue {}'.format(i))
        energies_ax.fill_between(x, av-stdev/2, av+stdev/2, color=c, alpha=.5)
    energies_ax.legend()
    energies_fig.savefig('{}/energies'.format(save_dir, save_dir))

    fig, ax = plt.subplots()
    for i in range(n_eigenfuncs):
        ax.plot(energies_array[-500:, i], label='Eigenvalue {}'.format(i))
    ax.legend()
    fig.savefig('{}/energies_newest'.format(save_dir, save_dir))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(loss)
    fig.savefig('{}/loss'.format(save_dir))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(loss[-500:])
    fig.savefig('{}/loss_newest'.format(save_dir))
    plt.close(fig)

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
