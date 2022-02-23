import jax
import jax.numpy as jnp                # JAX NumPy
from jax import grad

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train weight_dict

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
from backbone import EigenNet
from physics import hamiltonian_operator
from helper import moving_average


def create_train_state(n_dense_neurons, n_eigenfuncs, batch_size, D, learning_rate, decay_rate, sparsifying_K, n_space_dimension=2, init_rng=0):
    model = EigenNet([n_dense_neurons, n_dense_neurons, n_dense_neurons, n_eigenfuncs])
    batch = jnp.ones((batch_size, n_space_dimension))
    weight_dict = model.init(init_rng, (batch, D))
    layer_sparsifying_masks = EigenNet().get_all_layer_sparsifying_masks(weight_dict, sparsifying_K)
    weight_dict = EigenNet().sparsify_weights(weight_dict, layer_sparsifying_masks)

    """Creates initial `TrainState`."""
    tx = optax.rmsprop(learning_rate, decay_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=weight_dict, tx=tx), layer_sparsifying_masks


def get_network_as_function_of_input(params):
    return lambda x: EigenNet().apply(params, x)

def get_network_as_function_of_weights(batch):
    return lambda weights: EigenNet().apply(weights, x)

#@jax.jit
def train_step(state, batch, sigma_t_bar, j_sigma_t_bar, moving_average_beta):
    u = get_network_as_function_of_input(state.params)
    pred = u(batch)

    sigma_t_hat = jnp.mean(pred[:,:,None]@pred[:,:,None].swapaxes(2,1), axis=0)

    h_u = hamiltonian_operator(u, batch, system='hydrogen')
    pi_t_hat = jnp.mean(h_u[:,:,None]@pred[:,:,None].swapaxes(2,1), axis=0)

    sigma_t_bar = moving_average(sigma_t_bar, sigma_t_hat, beta=moving_average_beta)

    L = jnp.linalg.cholesky(sigma_t_bar)
    L_inv = jnp.linalg.inv(L)
    L_inv_T = L_inv.T
    L_diag_inv = jnp.linalg.inv(jnp.diag(L))

    u = get_network_as_function_of_weights(batch)
    del_u_del_weights = grad(u)

    j_pi_t_hat = h_u.T @ L_inv_T @ L_diag_inv @ del_u_del_weights
    j_pi_t_hat = jnp.mean(j_pi_t_hat, axis=0)
    Lambda = L_inv @ pi_t_hat @ L_inv_T
    j_sigma_t_hat = L_inv_T @ jnp.triu(Lambda @ jnp.linalg.inv(jnp.diag(L)))
    j_sigma_t_hat = jnp.mean(j_sigma_t_hat, axis=0)
    j_sigma_t_bar = moving_average(j_sigma_t_bar, j_sigma_t_hat, moving_average_beta)

    masked_grad = j_pi_t_hat - j_sigma_t_hat

    state = state.apply_gradients(grads=masked_grad)


    return state, energies, sigma_t_bar, j_sigma_t_bar



if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # Hyperparameter
    # Network parameter
    sparsifying_K = 3
    n_dense_neurons = 128
    n_eigenfuncs = 9

    # Optimizer
    learning_rate = 1e-5
    decay_rate = 0.999
    moving_average_beta = 0.01

    # Train setup
    num_epochs = 10000
    batch_size = 100

    # Simulation size
    D = 50

    # Create initial state
    state, layer_sparsifying_masks = create_train_state(n_dense_neurons, n_eigenfuncs, batch_size, D, learning_rate, decay_rate, sparsifying_K, init_rng=init_rng)
    sigma_t_bar = jnp.eye(n_eigenfuncs)
    j_sigma_t_bar = jnp.zeros((n_eigenfuncs, n_eigenfuncs))


    for epoch in range(1, num_epochs + 1):
      batch = np.random.uniform(0,1, size=(batch_size, 2))

      # Run an optimization step over a training batch
      state, energies, sigma_t_bar, j_sigma_t_bar = train_step(state, (batch, D), sigma_t_bar, j_sigma_t_bar, moving_average_beta)
      state.params = EigenNet().sparsify_weights(state.params, layer_sparsifying_masks)

