import jax
import jax.numpy as jnp                # JAX NumPy
from jax.config import config
config.update('jax_platform_name', 'cpu')
# config.update("jax_debug_nans", True)

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers

import helper
from tfn_backbone import TFNEigenNet
from backbone import EigenNet
from physics import construct_hamiltonian_function
from helper import moving_average
from flax.core import FrozenDict
import time
from tqdm import tqdm
from pathlib import Path
from flax.training import checkpoints
from jax import jit

from functools import partial
from jax import custom_jvp, custom_vjp

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.87'
import matplotlib.pyplot as plt

debug = False
# debug = True
if debug:
    jax.disable_jit()
#config.update("jax_enable_x64", True)

# jax.disable_jit()

EPSILON = 1e-8

system = 'hydrogen'
# system = 'laplace'
n_space_dimension = 3
n_electrons = 1
charge = 1
n_dense_neurons = [64]
n_eigenfuncs = 5

D_min = -15
D_max = 15

# rbf_low = 0.0
# rbf_high = D_max  / 2
# rbf_count = 8
# radial_channels = [

learning_rate = 1e-6
decay_rate = .999
moving_average_beta = 0.05
num_epochs = 300000
batch_size = 256

realtime_plots = True
n_plotting = 3000
log_every = 2000
window = 1000
save_dir = './results/{}_{}d_tfn_new'.format(system, n_space_dimension)


def create_TFN_train_state(init_rng=jax.random.PRNGKey(1)):
    model = TFNEigenNet(features=n_dense_neurons + [n_eigenfuncs], D_min=D_min, D_max=D_max,
    # rbf_low=rbf_low, rbf_high=rbf_high, rbf_count=rbf_count)
    # radial_channels=radial_channels)
    )
    batch = jnp.ones((n_electrons, n_space_dimension))
    weight_dict = model.init(init_rng, batch)

    vmodel = jax.vmap(model.apply, in_axes=[None, 0])

    @jit
    def apply_model(params, x, L_inv=None):
        x = x.reshape((-1, n_electrons, 3))
        out = vmodel(params, x)
        if L_inv is None:
            return out
        else:
            return jnp.einsum('ij, bj -> bi', L_inv, out)
    
    opt = optax.rmsprop(learning_rate, decay_rate)
    opt_state = opt.init(weight_dict)
    return apply_model, weight_dict, opt, opt_state

# def create_feedforward_train_state(init_rng=jax.random.PRNGKey(1)):
#     model = EigenNet(features=n_dense_neurons + [n_eigenfuncs], D_min=D_min, D_max=D_max)
#     batch = jnp.ones((batch_size, n_space_dimension))
#     weight_dict = model.init(init_rng, batch)
#     layer_sparsifying_masks = EigenNet.get_all_layer_sparsifying_masks(weight_dict, sparsifying_K)

#     opt = optax.rmsprop(learning_rate, decay_rate)
#     opt_state = opt.init(weight_dict)

#     @jit
#     def apply_model(params, x, L_inv=None):
#         out = model.apply(params, x)
#         if L_inv is None:
#             return out
#         else:
#             return jnp.einsum('ij, bj -> bi', L_inv, out)
#     return apply_model, weight_dict, opt, opt_state, layer_sparsifying_masks

@custom_vjp
def covariance(u1, u2):
    return jnp.mean(u1[:, :, None] @ u2[:, :, None].swapaxes(2, 1), axis=0)

def covariance_fwd(u1, u2):
    return covariance(u1, u2), (u1, u2, u1.shape[0])

def covariance_bwd(res, g):
    u1, u2, batch_size = res
    return ((u2 @ g)/ batch_size, (u1 @ g)/ batch_size)

covariance.defvjp(covariance_fwd, covariance_bwd)

# @partial(jit, static_argnums=(0,1,3,5))
def train_step(model_apply_jitted, h_fn, weight_dict, opt_update, opt_state, optax_apply_updates, batch, sigma_t_bar, j_sigma_t_bar):
    def u_from_theta(theta):
        return model_apply_jitted(theta, batch)

    def sigma_from_theta(theta):
        u = u_from_theta(theta)
        return covariance(u, u), u

    def pi_from_theta(theta):
        u = u_from_theta(theta)
        h_u = h_fn(theta, batch, u)
        return covariance(u, h_u), h_u
    
    j_sigma_t_hat, u = jax.jacrev(sigma_from_theta, has_aux=True)(weight_dict)
    j_sigma_t_bar = jax.tree_map(
        lambda x, y: moving_average(x, y, moving_average_beta),
        j_sigma_t_bar, j_sigma_t_hat
    )

    sigma = covariance(u, u)
    sigma_t_bar = moving_average(sigma_t_bar, sigma, moving_average_beta)

    # L = jnp.linalg.cholesky(sigma_t_bar + EPSILON*jnp.eye(sigma.shape[0]))
    L = jnp.linalg.cholesky(sigma_t_bar)
    L_inv = jnp.linalg.inv(L)

    pi, f_vjp, h_u = jax.vjp(pi_from_theta, weight_dict, has_aux=True)

    A_1_J_pi, = f_vjp(L_inv.T @ jnp.diag(jnp.diag(L_inv)))

    Lambda = L_inv @ pi @ L_inv.T

    energies = jnp.diag(Lambda)

    loss = jnp.sum(energies)

    A_2 = -L_inv.T @ jnp.triu(Lambda @ jnp.diag(jnp.diag(L_inv)))
    
    gradients = jax.tree_map(lambda sig_jac, loss_pi_grad: jnp.tensordot(A_2, sig_jac, [[0,1],[0,1]]) + loss_pi_grad, j_sigma_t_bar, A_1_J_pi)

    updates, opt_state = opt_update(gradients, opt_state)
    weight_dict = optax_apply_updates(weight_dict, updates)
    return loss, weight_dict, energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state

class ModelTrainer:

    def start_training(self, show_progress=True, callback=None):
        """
        Function for training the model
        """
        rng = jax.random.PRNGKey(1)
        rng, init_rng = jax.random.split(rng)
        # Create initial state
        model_apply_jitted, weight_dict, opt, opt_state = create_TFN_train_state(init_rng=init_rng)

        # Initialize sigma_t_bar as an identity matrix
        sigma_t_bar = jnp.eye(n_eigenfuncs)
        j_sigma_t_bar = jax.tree_map(lambda x: jnp.zeros((n_eigenfuncs, n_eigenfuncs) + x.shape), weight_dict)
        # j_sigma_t_bar = j_sigma_t_bar.unfreeze()
        start_epoch = 0
        loss = []
        energies = []

        h_fn = jit(construct_hamiltonian_function(model_apply_jitted, system=system, eps=0.0))
        opt_update_jitted = jit(lambda masked_gradient, opt_state: opt.update(masked_gradient, opt_state))
        optax_apply_updates_jitted = jit(lambda weight_dict, updates: optax.apply_updates(weight_dict, updates))

        if Path(save_dir).is_dir():
            weight_dict, opt_state, start_epoch, sigma_t_bar, j_sigma_t_bar = checkpoints.restore_checkpoint('{}/checkpoints/'.format(save_dir), (weight_dict, opt_state, start_epoch, sigma_t_bar, j_sigma_t_bar))
            loss, energies = np.load('{}/loss.npy'.format(save_dir)).tolist(), np.load('{}/energies.npy'.format(save_dir)).tolist()

        if realtime_plots:
            plt.ion()
        plots = helper.create_plots(n_space_dimension, n_eigenfuncs)

        # if debug:
        #     import pickle
        #     weights = pickle.load(open('weights.pkl', 'rb'))
        #     biases = pickle.load(open('biases.pkl', 'rb'))

        #     weight_dict = weight_dict.unfreeze()
        #     for i, key in enumerate(weight_dict['params'].keys()):
        #         weight_dict['params'][key]['kernel'] = weights[i]
        #         weight_dict['params'][key]['bias'] = biases[i]
        #     weight_dict = FrozenDict(weight_dict)

        pbar = tqdm(range(start_epoch+1, start_epoch+num_epochs+1), disable=not show_progress)
        for epoch in pbar:
            if debug:
                batch = jnp.array([[.3, .2], [.3, .4], [.9, .3]])
            else:
                # Generate a random batch
                rng, subkey = jax.random.split(rng)
                batch = jax.random.uniform(subkey, minval=D_min, maxval=D_max, shape=(batch_size, n_space_dimension))

            # Run an optimization step over a training batch
            new_loss, weight_dict, new_energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state = train_step(model_apply_jitted, h_fn, weight_dict, opt_update_jitted, opt_state, optax_apply_updates_jitted, batch, sigma_t_bar, j_sigma_t_bar)
            pbar.set_description('Loss {:.3f}'.format(np.around(np.asarray(new_loss), 3).item()))

            loss.append(new_loss)
            energies.append(new_energies)

            # Run a callback function to decide whether to stop training
            if callback is not None:
                to_stop = callback(epoch, energies=energies)
                if to_stop == True:
                    return

            # Save a check point
            if epoch % log_every == 0 or epoch == 1:
                helper.create_checkpoint(save_dir, model_apply_jitted, weight_dict, D_min, D_max, n_space_dimension, opt_state, epoch, sigma_t_bar, j_sigma_t_bar, loss, energies, n_eigenfuncs, charge, system, L_inv, window, n_plotting, *plots)
                plt.pause(.01)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training()





