import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
from backbone import EigenNet


def cross_entropy_loss(*, logits, labels):
  one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
  return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    net = EigenNet()
    params = net.init(rng, jnp.ones([1, 2]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
  def loss_fn(params):
    logits = EigenNet().apply({'params': params}, batch['cooridnates'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    return loss, logits

  print(jax.device_get(batch))
  print(jax.device_get(EigenNet().apply({'params': state.params}, batch)))
  exit()

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=batch['label'])
  return state, metrics


def train_epoch(state, coordinate_batch):

  state, energies = train_step(state, coordinate_batch)
  energies = jax.device_get(energies)

  return state, energies



if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    learning_rate = 0.1
    momentum = 0.9

    state = create_train_state(init_rng, learning_rate, momentum)

    num_epochs = 10000
    batch_size = 32

    for epoch in range(1, num_epochs + 1):
      coordinate_batch = np.random.uniform(0,1, size=(batch_size, 2))

      # Run an optimization step over a training batch
      state, energies = train_epoch(state, coordinate_batch)
      # Evaluate on the test set after each training epoch

