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
#from jax.example_libraries import optimizers
from functools import partial
from helper import vectorized_hessian
from jax import vmap

import matplotlib.pyplot as plt

debug = False
debug = True
if debug:
    jax.disable_jit()
# config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
# config.update("jax_debug_nans", True)

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
        masked_grad = (j_pi_t_hat - j_sigma_t_bar)
        return masked_grad

    return _calculate_j_pi_t_hat, _calculate_j_sigma_t_bar, _calculate_masked_gradient

# This jit seems not making any difference
# def calculate_masked_gradient(del_u_del_weights, pred, h_u, sigma_t_bar, moving_average_beta, j_sigma_t_bar):
#     sigma_t_hat = np.mean(pred[:, :, None]@pred[:, :, None].swapaxes(2, 1), axis=0)
#     pi_t_hat = np.mean(pred[:, :, None]@h_u[:, :, None].swapaxes(2, 1), axis=0)

#     sigma_t_bar = moving_average(sigma_t_bar, sigma_t_hat, beta=moving_average_beta)

#     def sigma_from_weights(weights):
#         return np.mean(pred[:, :, None]@pred[:, :, None].swapaxes(2, 1), axis=0)
#     j_sigma_t_hat = jacrev()
#     def loss_from_sigma(sigma):
#         L = jnp.linalg.cholesky(sigma_t_bar)
#         L_inv = jnp.linalg.inv(L)
#         Lambda = L_inv @ pi_t_hat @ L_inv.T
#         return jnp.trace(Lambda)
    
#     L = jnp.linalg.cholesky(sigma_t_bar)
#     L_inv = jnp.linalg.inv(L)
#     L_inv_T = L_inv.T
#     L_diag_inv = jnp.eye(L.shape[0]) * (1/jnp.diag(L))
    
#     A_1 = L_inv_T @ L_diag_inv
#     A_1 = h_u @ A_1
    
#     Lambda = L_inv @ pi_t_hat @ L_inv_T
#     A_2 = L_inv_T @ jnp.triu(Lambda @ L_diag_inv)
#     A_2 = pred @ A_2
    
#     _calculate_j_pi_t_hat, _calculate_j_sigma_t_bar, _calculate_masked_gradient = get_masked_gradient_function(A_1, A_2, moving_average_beta)
#     j_pi_t_hat = jax.tree_multimap(_calculate_j_pi_t_hat, del_u_del_weights)
#     j_sigma_t_bar = jax.tree_multimap(_calculate_j_sigma_t_bar, del_u_del_weights, j_sigma_t_bar)

#     loss, sigma_back = jax.value_and_grad(loss_from_sigma)(sigma_t_bar)

#     del_u_del_weights = jax.tree_multimap(_calculate_masked_gradient, j_pi_t_hat, j_sigma_t_bar)
    
#     masked_grad = jax.tree_multimap(lambda sig_jac, grad: jnp.tensordot(sigma_back, sig_jac, [[0,1],[0,1]]) + grad,
#     j_sigma_t_bar, del_u_del_weights)

#     return FrozenDict(masked_grad), Lambda, L_inv, j_sigma_t_bar

def train_step(model_apply_jitted, del_u_del_weights_fn, h_fn, weight_dict, opt_update, opt_state, optax_apply_updates, batch, sigma_t_bar, j_sigma_t_bar, moving_average_beta):
    # pred = model_apply_jitted(weight_dict, batch)

    # del_u_del_weights = del_u_del_weights_fn(weight_dict, batch)

    # h_u = h_fn(weight_dict, batch, pred)
    # masked_gradient, Lambda, L_inv, j_sigma_t_bar = calculate_masked_gradient(del_u_del_weights, pred, h_u, sigma_t_bar, moving_average_beta, j_sigma_t_bar)

    # weight_dict = FrozenDict(weight_dict)
    # updates, opt_state = opt_update(masked_gradient, opt_state)
    # weight_dict = optax_apply_updates(weight_dict, updates)

    # loss = jnp.trace(Lambda)
    # energies = jnp.diag(Lambda)

    # return loss, weight_dict, energies, sigma_t_bar, j_sigma_t_bar, L_inv, opt_state

    def covariance(x, y):
        return jnp.mean(x[:, :, None]@y[:, :, None].swapaxes(2, 1), axis=0)

    def u_from_theta(theta):
        return model_apply_jitted(theta, batch)

    def sigma_from_theta(theta):
        u = u_from_theta(theta)
        return covariance(u, u), u
    
    def sigma_avg_from_theta(theta):
        sigma_hat, u = sigma_from_theta(theta)
        return moving_average(jax.lax.stop_gradient(sigma_t_bar), sigma_hat, moving_average_beta), u

    j_sigma_t_hat, u = jax.jacrev(sigma_from_theta, has_aux=True)(weight_dict)
    j_sigma_t_bar = jax.tree_multimap(
        lambda x, y: moving_average(x, y, moving_average_beta),
        j_sigma_t_bar, j_sigma_t_hat
    )
    
    def loss_from_sigma(sigma, u):
        h_u = h_fn(weight_dict, batch, u)
        pi = covariance(u, h_u)
        L = jnp.linalg.cholesky(sigma)
        L_inv = jnp.linalg.inv(L)
        Lambda = L_inv @ pi @ L_inv.T
        eigval = eigenvalues(sigma, pi)
        return jnp.sum(eigval), (eigval, L_inv, Lambda)

    def loss_from_theta(theta):
        sigma_avg, u = sigma_avg_from_theta(theta)
        loss, aux = loss_from_sigma(sigma_avg, u)
        return loss, aux + (sigma_avg, u)

    @jax.custom_vjp
    def eigenvalues(sigma, pi):
        return eigenvalues_fwd(sigma, pi)[0]
    
    def eigenvalues_fwd(sigma, pi):
        chol = jnp.linalg.cholesky(sigma)
        choli = jnp.linalg.inv(chol)
        rq = choli @ pi @ choli.T
        eigval = jnp.diag(rq)
        dl = jnp.diag(jnp.diag(choli))
        triu = jnp.triu(rq @ dl)
        dsigma = -(choli.T @ triu)
        dpi = choli.T @ dl
        return eigval, (dsigma, dpi)
    
    def eigenvalues_bwd(res, g):
        dsigma, dpi = res
        return dsigma * g, dpi * g
    
    eigenvalues.defvjp(eigenvalues_fwd, eigenvalues_bwd)

        
    val, loss_back = jax.value_and_grad(loss_from_theta, has_aux=True)(weight_dict)
    loss, aux = val
    energies, L_inv, Lambda, sigma_t_bar, u = aux
    sigma_back, _ = jax.grad(loss_from_sigma, has_aux=True)(sigma_t_bar, u)

    gradients = jax.tree_multimap(lambda sig_jac, loss_grad: jnp.tensordot(sigma_back, sig_jac, [[0,1],[0,1]]) + loss_grad, j_sigma_t_bar, loss_back)

    weight_dict = FrozenDict(weight_dict)
    gradients = FrozenDict(gradients)
    updates, opt_state = opt_update(gradients, opt_state)
    # weight_dict = weight_dict.unfreeze()
    # gradients = gradients.unfreeze()
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
        self.sparsifying_K = 0
        self.n_dense_neurons = [128, 128, 128]
        self.n_eigenfuncs = 4

        # Turn on/off real time plotting
        self.realtime_plots = True
        self.npts = 64
        self.log_every = 1000
        self.window = 100

        # Optimizer
        self.learning_rate = 1e-5
        self.decay_rate = 0.999
        self.moving_average_beta = 0.01

        # Train setup
        self.num_epochs = 100000
        self.batch_size = 128
        self.save_dir = './results/{}_{}d'.format(self.system, self.n_space_dimension)

        # Simulation size
        self.D_min = -50
        self.D_max = 50
        if (self.system, self.n_space_dimension) == ('hydrogen', 2):
            self.D_min = -50
            self.D_max = 50

    def start_training(self, show_progress = True, callback = None):
        rng = jax.random.PRNGKey(1)
        rng, init_rng = jax.random.split(rng)
        # Create initial state
        model, weight_dict, opt, opt_state, layer_sparsifying_masks = create_train_state(self.n_dense_neurons, self.n_eigenfuncs, self.batch_size, self.D_min, self.D_max, self.learning_rate, self.decay_rate, self.sparsifying_K, n_space_dimension=self.n_space_dimension, init_rng=init_rng)


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


        if debug:
            import pickle
            weights = pickle.load(open('weights.pkl', 'rb'))
            biases = pickle.load(open('biases.pkl', 'rb'))

            weight_dict = weight_dict.unfreeze()
            for i, key in enumerate(weight_dict['params'].keys()):
                weight_dict['params'][key]['kernel'] = weights[i]
                weight_dict['params'][key]['bias'] = biases[i]
            weight_dict = FrozenDict(weight_dict)
        pbar = tqdm(range(start_epoch+1, start_epoch+self.num_epochs+1),disable = not show_progress)
        for epoch in pbar:
            if debug:
                batch = jnp.array([[.3, .2], [.3, .4], [.9, .3]])
            # else:
            # rng, subkey = jax.random.split(rng)
            # batch = jax.random.uniform(subkey, minval=self.D_min, maxval=self.D_max, shape=(self.batch_size, self.n_space_dimension))

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
                helper.create_checkpoint(self.save_dir, model, weight_dict, self.D_min, self.D_max, self.n_space_dimension, opt_state, epoch, sigma_t_bar, j_sigma_t_bar, loss, energies, self.n_eigenfuncs, self.charge, self.system, L_inv, self.window, *plots)
                plt.pause(.01)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training()





