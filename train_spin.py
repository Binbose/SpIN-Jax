from time import sleep
import jax
import jax.numpy as jnp                # JAX NumPy
import matplotlib.pyplot as plt
from jax import grad, jacfwd, jacrev

from flax import linen as nn           # The Linen API
# Useful dataclass to keep train weight_dict
from flax.training import train_state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers

import helper
from backbone import EigenNet
from physics import hamiltonian_operator
from helper import moving_average
from flax.core import FrozenDict
import time
from tqdm import tqdm
from pathlib import Path
from flax.training import checkpoints
from jax import jit

def create_train_state(n_dense_neurons, n_eigenfuncs, batch_size, D, learning_rate, decay_rate, sparsifying_K, n_space_dimension=2, init_rng=0):
    model = EigenNet(features=n_dense_neurons + [n_eigenfuncs], D=D)
    batch = jnp.ones((batch_size, n_space_dimension))
    weight_dict = model.init(init_rng, batch)
    layer_sparsifying_masks = EigenNet.get_all_layer_sparsifying_masks(weight_dict, sparsifying_K)

    """Creates initial `TrainState`."""
    opt = optax.rmsprop(learning_rate, decay_rate)
    opt_state = opt.init(weight_dict)
    return model, weight_dict, opt, opt_state, layer_sparsifying_masks


def get_network_as_function_of_input(model_apply, params):
    return lambda batch: model_apply(params, batch)


def get_network_as_function_of_weights(model_apply, batch):
    return lambda weights: model_apply(weights, batch)

# This jit seems not making any difference
def calculate_masked_gradient(del_u_del_weights, pred, h_u, sigma_t_bar, moving_average_beta):
    sigma_t_hat = jnp.mean(pred[:, :, None]@pred[:, :, None].swapaxes(2, 1), axis=0)
    pi_t_hat = jnp.mean(h_u[:, :, None]@pred[:, :, None].swapaxes(2, 1), axis=0)

    sigma_t_bar = moving_average(sigma_t_bar, sigma_t_hat, beta=moving_average_beta)

    L = jnp.linalg.cholesky(sigma_t_bar)
    L_inv = jnp.linalg.inv(L)
    L_inv_T = L_inv.T
    L_diag_inv = jnp.eye(L.shape[0]) * (1/jnp.diag(L))

    A_1 = L_inv_T @ L_diag_inv
    A_1 = h_u @ A_1

    Lambda = L_inv @ pi_t_hat @ L_inv_T
    A_2 = L_inv_T @ jnp.triu(Lambda @ L_diag_inv)
    A_2 = pred @ A_2

    for key in del_u_del_weights['params'].keys():
        j_pi_t_hat = jnp.einsum('bj, bjcw -> bcw', A_1,del_u_del_weights['params'][key]['kernel'])
        j_pi_t_hat = jnp.mean(j_pi_t_hat, axis=0)

        j_sigma_t_hat = jnp.einsum('bj, bjcw -> bcw', A_2,  del_u_del_weights['params'][key]['kernel'])
        j_sigma_t_hat = jnp.mean(j_sigma_t_hat, axis=0)
        j_sigma_t_bar[key] = moving_average(j_sigma_t_bar[key], j_sigma_t_hat, moving_average_beta)

        masked_grad = j_pi_t_hat - j_sigma_t_bar[key]
        del_u_del_weights['params'][key]['kernel'] = masked_grad

    return FrozenDict(del_u_del_weights), Lambda, L_inv


def train_step(model_apply_jitted, weight_dict, opt, opt_state, batch, sigma_t_bar, j_sigma_t_bar, moving_average_beta):
    u_of_x = get_network_as_function_of_input(model_apply_jitted, weight_dict)
    u_of_w = get_network_as_function_of_weights(model_apply_jitted, batch)

    pred = u_of_x(batch)
    del_u_del_weights = jacrev(u_of_w)(weight_dict)

    h_u = hamiltonian_operator(u_of_x, batch, fn_x=pred, system=system, nummerical_diff=False, eps=0.01)
    print(h_u.sum(0))
    exit()
    masked_gradient, Lambda, L_inv = calculate_masked_gradient(del_u_del_weights, pred, h_u, sigma_t_bar, moving_average_beta)


    weight_dict = FrozenDict(weight_dict)
    updates, opt_state = opt.update(masked_gradient, opt_state)
    weight_dict = optax.apply_updates(weight_dict, updates)

    loss = jnp.trace(Lambda)
    energies = jnp.diag(Lambda)

    return loss, weight_dict, energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state


if __name__ == '__main__':
    rng = jax.random.PRNGKey(8)
    rng, init_rng = jax.random.split(rng)

    # Problem definition
    #system = 'hydrogen'
    system = 'laplace'
    n_space_dimension = 1

    # Hyperparameter
    # Network parameter
    sparsifying_K = 3
    n_dense_neurons = [64, 64, 64, 32]
    n_eigenfuncs = 4


    # Optimizer
    learning_rate = 1e-4
    decay_rate = 0.999
    moving_average_beta = 1

    # Train setup
    num_epochs = 5000
    batch_size = 128
    save_dir = './results/{}'.format(system)

    # Simulation size
    D = np.pi

    # Create initial state
    model, weight_dict, opt, opt_state, layer_sparsifying_masks = create_train_state(n_dense_neurons, n_eigenfuncs, batch_size, D, learning_rate, decay_rate, sparsifying_K, n_space_dimension=n_space_dimension, init_rng=init_rng)
    weight_dict = weight_dict.unfreeze()

    weight_list = np.load('./weights.npy', allow_pickle=True)

    for i, key in enumerate(weight_dict['params'].keys()):
        w, b = weight_list[i]
        weight_dict['params'][key]['kernel'] = w

    sigma_t_bar = jnp.eye(n_eigenfuncs)
    j_sigma_t_bar = {key: jnp.zeros_like(weight_dict['params'][key]['kernel']) for key in weight_dict['params'].keys()}
    start_epoch = 0
    loss = []
    energies = []

    #model_apply_jitted = jax.jit(lambda params, inputs: model.apply(params, inputs))
    model_apply_jitted = lambda params, inputs: model.apply(params, inputs)

    if Path(save_dir).is_dir():
        weight_dict, opt_state, start_epoch, sigma_t_bar, j_sigma_t_bar = checkpoints.restore_checkpoint('{}/checkpoints/'.format(save_dir), (weight_dict, opt_state, start_epoch, sigma_t_bar, j_sigma_t_bar))
        loss, energies = np.load('{}/loss.npy'.format(save_dir)).tolist(), np.load('{}/energies.npy'.format(save_dir)).tolist()

    pbar = tqdm(range(start_epoch+1, start_epoch+num_epochs+1))
    for epoch in pbar:
        #batch = jax.random.uniform(rng, minval=0, maxval=D, shape=(batch_size, n_space_dimension))
        batch = np.load('./batch.npy')
        # Run an optimization step over a training batch
        new_loss, weight_dict, new_energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state = train_step(model_apply_jitted, weight_dict, opt, opt_state, batch, sigma_t_bar, j_sigma_t_bar, moving_average_beta)
        pbar.set_description('Loss {:.2f}'.format(np.around(np.asarray(new_loss), 3).item()))

        '''
        weight_dict = EigenNet.sparsify_weights(
            weight_dict, layer_sparsifying_masks)
        '''
        weight_dict = weight_dict.unfreeze()


        loss.append(new_loss)
        energies.append(new_energies)


        if epoch % 1000 == 0:
            helper.create_checkpoint(save_dir, opt_state, epoch, sigma_t_bar, j_sigma_t_bar, loss, energies, n_eigenfuncs, L_inv)





