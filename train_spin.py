import jax
import jax.numpy as jnp                # JAX NumPy
from jax import grad, jacfwd, jacrev

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers

import helper
from backbone import EigenNet
from physics import construct_hamiltonian_function
from helper import moving_average
from flax.core import FrozenDict
import time
from tqdm import tqdm
from pathlib import Path
from flax.training import checkpoints
from jax import jit
from jax.config import config
from functools import partial
from jax import custom_jvp, custom_vjp


import matplotlib.pyplot as plt


#config.update("jax_enable_x64", True)
#config.update('jax_platform_name', 'cpu')
# config.update("jax_debug_nans", True)


def create_train_state(n_dense_neurons, n_eigenfuncs, batch_size, D_min, D_max, learning_rate, decay_rate, sparsifying_K, n_space_dimension=2, init_rng=0):
    model = EigenNet(features=n_dense_neurons + [n_eigenfuncs], D_min=D_min, D_max=D_max)
    batch = jnp.ones((batch_size, n_space_dimension))
    weight_dict = model.init(init_rng, batch)
    layer_sparsifying_masks = EigenNet.get_all_layer_sparsifying_masks(weight_dict, sparsifying_K)

    opt = optax.rmsprop(learning_rate, decay_rate)
    opt_state = opt.init(weight_dict)
    return model, weight_dict, opt, opt_state, layer_sparsifying_masks

@custom_vjp
def covariance(u1, u2):
    return jnp.mean(u1[:, :, None] @ u2[:, :, None].swapaxes(2, 1), axis=0)

def covariance_fwd(u1, u2):
    return covariance(u1, u2), (u1, u2, u1.shape[0])

def covariance_bwd(res, g):
    u1, u2, batch_size = res
    return ((u2 @ g)/ batch_size, (u1 @ g)/ batch_size)

covariance.defvjp(covariance_fwd, covariance_bwd)

@partial(jit, static_argnums=(0,1,3,5))
def train_step(model_apply_jitted, h_fn, weight_dict, opt_update, opt_state, optax_apply_updates, batch, sigma_t_bar, j_sigma_t_bar, moving_average_beta):
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
    j_sigma_t_bar = jax.tree_multimap(
        lambda x, y: moving_average(x, y, moving_average_beta),
        j_sigma_t_bar, j_sigma_t_hat
    )

    sigma = covariance(u, u)
    sigma_t_bar = moving_average(sigma_t_bar, sigma, moving_average_beta)

    L = jnp.linalg.cholesky(sigma_t_bar)
    L_inv = jnp.linalg.inv(L)

    pi, f_vjp, h_u = jax.vjp(pi_from_theta, weight_dict, has_aux=True)

    A_1_J_pi, = f_vjp(L_inv.T @ jnp.diag(jnp.diag(L_inv)))

    Lambda = L_inv @ pi @ L_inv.T

    energies = jnp.diag(Lambda)

    loss = jnp.sum(energies)

    A_2 = -L_inv.T @ jnp.triu(Lambda @ jnp.diag(jnp.diag(L_inv)))
    
    gradients = jax.tree_multimap(lambda sig_jac, loss_pi_grad: jnp.tensordot(A_2, sig_jac, [[0,1],[0,1]]) + loss_pi_grad, j_sigma_t_bar, A_1_J_pi)

    weight_dict = FrozenDict(weight_dict)
    gradients = FrozenDict(gradients)
    updates, opt_state = opt_update(gradients, opt_state)
    weight_dict = optax_apply_updates(weight_dict, updates)
    weight_dict = FrozenDict(weight_dict)
    return loss, weight_dict, energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state



class ModelTrainer:
    def __init__(self) -> None:
        # Hyperparameter
        # Problem definition
        self.system = 'hydrogen'
        # self.system = 'laplace'
        self.n_space_dimension = 2
        self.charge = 1

        # Network parameter
        self.sparsifying_K = 5
        self.n_dense_neurons = [128, 128, 128]
        self.n_eigenfuncs = 4

        # Turn on/off real time plotting
        self.realtime_plots = True
        self.n_plotting = 200
        self.log_every = 10000
        self.window = 1000

        # Optimizer
        self.learning_rate = 1e-5
        self.decay_rate = 0.999
        self.moving_average_beta = 0.01

        # Train setup
        self.num_epochs = 200000
        self.batch_size = 128
        self.save_dir = './results/{}_{}d'.format(self.system, self.n_space_dimension)

        # Simulation size
        self.D_min = -50
        self.D_max = 50
        if (self.system, self.n_space_dimension) == ('hydrogen', 2):
            self.D_min = -25
            self.D_max = 25

    def start_training(self, show_progress=True, callback=None):
        """
        Function for training the model
        """
        rng = jax.random.PRNGKey(1)
        rng, init_rng = jax.random.split(rng)
        # Create initial state
        model, weight_dict, opt, opt_state, layer_sparsifying_masks = create_train_state(self.n_dense_neurons, self.n_eigenfuncs, self.batch_size, self.D_min, self.D_max, self.learning_rate, self.decay_rate, self.sparsifying_K, n_space_dimension=self.n_space_dimension, init_rng=init_rng)

        # Initialize sigma_t_bar as an identity matrix
        sigma_t_bar = jnp.eye(self.n_eigenfuncs)
        j_sigma_t_bar = jax.tree_multimap(lambda x: jnp.zeros((self.n_eigenfuncs, self.n_eigenfuncs) + x.shape), weight_dict).unfreeze()
        start_epoch = 0
        loss = []
        energies = []


        model_apply_jitted = jit(lambda params, inputs: model.apply(params, inputs))
        h_fn = jit(construct_hamiltonian_function(model_apply_jitted, system=self.system, eps=0.0))
        del_u_del_weights_fn = jit(jacrev(model_apply_jitted, argnums=0))
        opt_update_jitted = jit(lambda masked_gradient, opt_state: opt.update(masked_gradient, opt_state))
        optax_apply_updates_jitted = jit(lambda weight_dict, updates: optax.apply_updates(weight_dict, updates))

        if Path(self.save_dir).is_dir():
            weight_dict, opt_state, start_epoch, sigma_t_bar, j_sigma_t_bar = checkpoints.restore_checkpoint('{}/checkpoints/'.format(self.save_dir), (weight_dict, opt_state, start_epoch, sigma_t_bar, j_sigma_t_bar))
            loss, energies = np.load('{}/loss.npy'.format(self.save_dir)).tolist(), np.load('{}/energies.npy'.format(self.save_dir)).tolist()

        if self.realtime_plots:
            plt.ion()
        plots = helper.create_plots(self.n_space_dimension, self.n_eigenfuncs)

        pbar = tqdm(range(start_epoch+1, start_epoch+self.num_epochs+1), disable=not show_progress)
        for epoch in pbar:
            # Generate a random batch
            rng, subkey = jax.random.split(rng)
            batch = jax.random.uniform(subkey, minval=self.D_min, maxval=self.D_max, shape=(self.batch_size, self.n_space_dimension))

            # Sparsify the weights so that some of them are always 0 (to check)
            if self.sparsifying_K > 0:
                weight_dict = EigenNet.sparsify_weights(weight_dict, layer_sparsifying_masks)

            weight_dict = weight_dict.unfreeze()

            # Run an optimization step over a training batch
            new_loss, weight_dict, new_energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state = train_step(model_apply_jitted, h_fn, weight_dict, opt_update_jitted, opt_state, optax_apply_updates_jitted, batch, sigma_t_bar, j_sigma_t_bar, self.moving_average_beta)
            pbar.set_description('Loss {:.3f}'.format(np.around(np.asarray(new_loss), 3).item()))

            loss.append(new_loss)
            energies.append(new_energies)

            # Run a callback function to decide whether to stop training
            if callback is not None:
                to_stop = callback(epoch, energies=energies)
                if to_stop == True:
                    return

            # Save a check point
            if epoch % self.log_every == 0 or epoch == 1:
                helper.create_checkpoint(self.save_dir, model, weight_dict, self.D_min, self.D_max, self.n_space_dimension, opt_state, epoch, sigma_t_bar, j_sigma_t_bar, loss, energies, self.n_eigenfuncs, self.charge, self.system, L_inv, self.window, self.n_plotting, *plots)
                plt.pause(.01)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training()





