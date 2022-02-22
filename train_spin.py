import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train weight_dict

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
from backbone import EigenNet



def create_train_state(batch_size, D, learning_rate, decay_rate, sparsifying_K, n_space_dimension=2, init_rng=0):
    model = EigenNet()
    batch = jnp.ones((batch_size, n_space_dimension))
    weight_dict = model.init(init_rng, (batch, D))
    layer_sparsifying_masks = EigenNet().get_all_layer_sparsifying_masks(weight_dict, sparsifying_K)
    weight_dict = EigenNet().sparsify_weights(weight_dict, layer_sparsifying_masks)

    """Creates initial `TrainState`."""
    tx = optax.rmsprop(learning_rate, decay_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=weight_dict, tx=tx), layer_sparsifying_masks


#@jax.jit
def train_step(state, batch):
    pred = EigenNet().apply(state.params, batch)

    sigma_t_hat = jnp.sum(pred[:,:,None]@pred[:,:,None].swapaxes(2,1), axis=0)
    pi_t_hat =


    '''
    def loss_fn(params):
    pred = EigenNet().apply({'params': params}, batch['cooridnates'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    return loss, logits
    
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state
'''


def train_epoch(state, batch):

    state = train_step(state, batch)

    return state, energies



if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # Hyperparameter
    # Network parameter
    sparsifying_K = 3

    # Optimizer
    learning_rate = 1e-5
    decay_rate = 0.999

    # Train setup
    num_epochs = 10000
    batch_size = 100

    # Simulation size
    D = 50

    # Create initial state
    state, layer_sparsifying_masks = create_train_state(batch_size, D, learning_rate, decay_rate, sparsifying_K, init_rng=init_rng)



    for epoch in range(1, num_epochs + 1):
      batch = np.random.uniform(0,1, size=(batch_size, 2))

      # Run an optimization step over a training batch
      state, energies = train_epoch(state, (batch, D))
      state.params = EigenNet().sparsify_weights(state.params, layer_sparsifying_masks)

