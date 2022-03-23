import jax
import jax.numpy as jnp                # JAX NumPy
from jax import grad, jacfwd, jacrev

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers

import helper
from backbone import EigenNet
from physics import hamiltonian_operator, construct_hamiltonian_function
from helper import moving_average
from flax.core import FrozenDict
import time
from tqdm import tqdm
from pathlib import Path
from flax.training import checkpoints
from jax import jit
from jax.config import config
#from jax.example_libraries import optimizers
from functools import partial
from helper import vectorized_hessian
from jax import vmap

import matplotlib.pyplot as plt

#config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)

def create_train_state(n_dense_neurons, n_eigenfuncs, batch_size, D_min, D_max, learning_rate, decay_rate, sparsifying_K, n_space_dimension=2, init_rng=0):
    model = EigenNet(features=n_dense_neurons + [n_eigenfuncs], D_min=D_min, D_max=D_max)
    batch = jnp.ones((batch_size, n_space_dimension))
    weight_dict = model.init(init_rng, batch)
    layer_sparsifying_masks = EigenNet.get_all_layer_sparsifying_masks(weight_dict, sparsifying_K)

    #opt = optax.rmsprop(learning_rate, decay_rate)
    opt = optax.adam(learning_rate, decay_rate)
    opt_state = opt.init(weight_dict)

    #opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    #opt_state = opt_init(weight_dict)
    return model, weight_dict, opt, opt_state, layer_sparsifying_masks

def get_network_as_function_of_input(model_apply, params):
    return lambda batch: model_apply(params, batch)

def get_network_as_function_of_weights(model_apply, batch):
    return lambda weights: model_apply(weights, batch)

def get_masked_gradient_function(A_1, A_2, moving_average_beta):
    def _calculate_j_pi_t_hat(del_u_del_weights):
        j_pi_t_hat = jnp.tensordot(A_1, del_u_del_weights, [[0, 1], [0, 1]])

        return j_pi_t_hat

    def _calculate_j_sigma_t_bar(del_u_del_weights, j_sigma_t_bar):
        j_sigma_t_hat = jnp.tensordot(A_2, del_u_del_weights, [[0, 1], [0, 1]])
        j_sigma_t_bar = moving_average(j_sigma_t_bar, j_sigma_t_hat, moving_average_beta)

        return j_sigma_t_bar

    def _calculate_masked_gradient(j_pi_t_hat, j_sigma_t_bar):
        masked_grad = -(j_pi_t_hat - j_sigma_t_bar)

        return masked_grad

    return _calculate_j_pi_t_hat, _calculate_j_sigma_t_bar, _calculate_masked_gradient




#@jit
def calculate_masked_gradient(del_u_del_weights, pred, h_u, sigma_t_bar, moving_average_beta, j_sigma_t_bar):
    sigma_t_hat = np.mean(pred[:, :, None]@pred[:, :, None].swapaxes(2, 1), axis=0)
    pi_t_hat = np.mean(pred[:, :, None]@h_u[:, :, None].swapaxes(2, 1), axis=0)

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
    
    _calculate_j_pi_t_hat, _calculate_j_sigma_t_bar, _calculate_masked_gradient = get_masked_gradient_function(A_1, A_2, moving_average_beta)
    j_pi_t_hat = jax.tree_multimap(_calculate_j_pi_t_hat, del_u_del_weights)
    j_sigma_t_bar = jax.tree_multimap(_calculate_j_sigma_t_bar, del_u_del_weights, j_sigma_t_bar)

    del_u_del_weights = jax.tree_multimap(_calculate_masked_gradient, j_pi_t_hat, j_sigma_t_bar)

    return FrozenDict(del_u_del_weights), Lambda, L_inv, j_sigma_t_bar

def train_step(model_apply_jitted, del_u_del_weights_fn, h_fn, weight_dict, opt_update, opt_state, optax_apply_updates, batch, sigma_t_bar, j_sigma_t_bar, moving_average_beta):
    pred = model_apply_jitted(weight_dict, batch)

    del_u_del_weights = del_u_del_weights_fn(weight_dict, batch)
    print(jax.tree_multimap(lambda x: x.shape, del_u_del_weights))
    exit()

    h_u = h_fn(weight_dict, batch, pred)

    masked_gradient, Lambda, L_inv, j_sigma_t_bar = calculate_masked_gradient(del_u_del_weights, pred, h_u, sigma_t_bar, moving_average_beta, j_sigma_t_bar)

    weight_dict = FrozenDict(weight_dict)
    updates, opt_state = opt_update(masked_gradient, opt_state)
    weight_dict = optax_apply_updates(weight_dict, updates)

    loss = jnp.trace(Lambda)
    energies = jnp.diag(Lambda)

    return loss, weight_dict, energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state

class ModelTrainer:
    def __init__(self) -> None:
        # Hyperparameter
        # Problem definition
        self.system = 'hydrogen'
        #self.system = 'laplace'
        self.n_space_dimension = 2
        self.charge = 1

        # Network parameter
        self.sparsifying_K = 0
        self.n_dense_neurons = [128, 128, 128]
        self.n_eigenfuncs = 4

        # Turn on/off real time plotting
        self.realtime_plots = True
        self.npts = 256
        self.log_every = 10000
        self.window = 100

        # Optimizer
        self.learning_rate = 1e-5
        self.decay_rate = 0.999
        self.moving_average_beta = 0.01

        # Train setup
        self.num_epochs = 100000
        self.batch_size = 256
        self.save_dir = './results/{}_{}d'.format(self.system, self.n_space_dimension)

        # Simulation size
        self.D_min = 0
        self.D_max = np.pi
        if (self.system, self.n_space_dimension) == ('hydrogen', 2):
            self.D_min = -50
            self.D_max = 50

    def start_training(self, show_progress=True, callback=None):
        rng = jax.random.PRNGKey(2)
        rng, init_rng = jax.random.split(rng)
        # Create initial state
        model, weight_dict, opt, opt_state, layer_sparsifying_masks = create_train_state(self.n_dense_neurons, self.n_eigenfuncs, self.batch_size, self.D_min, self.D_max, self.learning_rate, self.decay_rate, self.sparsifying_K, n_space_dimension=self.n_space_dimension, init_rng=init_rng)


        sigma_t_bar = jnp.eye(self.n_eigenfuncs)
        j_sigma_t_bar = jax.tree_multimap(lambda x: jnp.zeros_like(x), weight_dict).unfreeze()
        start_epoch = 0
        loss = []
        energies = []

        model_apply_jitted = jax.jit(lambda params, inputs: model.apply(params, inputs))
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


        pbar = tqdm(range(start_epoch+1, start_epoch+self.num_epochs+1),disable = not show_progress)
        for epoch in pbar:
            rng, sub_key = jax.random.split(rng)
            batch = jax.random.uniform(sub_key, minval=self.D_min, maxval=self.D_max, shape=(self.batch_size, self.n_space_dimension))

            if self.sparsifying_K > 0:
                weight_dict = EigenNet.sparsify_weights(weight_dict, layer_sparsifying_masks)

            weight_dict = weight_dict.unfreeze()
            # Run an optimization step over a training batch
            new_loss, weight_dict, new_energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state = train_step(model_apply_jitted, del_u_del_weights_fn, h_fn, weight_dict, opt_update_jitted, opt_state, optax_apply_updates_jitted, batch, sigma_t_bar, j_sigma_t_bar, self.moving_average_beta)
            pbar.set_description('Loss {:.3f}'.format(np.around(np.asarray(new_loss), 3).item()))

            loss.append(new_loss)
            energies.append(new_energies)

            if callback is not None:
                to_stop = callback(epoch, energies=energies)
                if to_stop == True:
                    return

            if epoch % self.log_every == 0 or epoch == 1:
                helper.create_checkpoint(self.save_dir, model, weight_dict, self.D_min, self.D_max, self.n_space_dimension, opt_state, epoch, sigma_t_bar, j_sigma_t_bar, loss, energies, self.n_eigenfuncs, self.charge, self.system, L_inv, self.window, self.npts, *plots)
                plt.pause(.01)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training()





