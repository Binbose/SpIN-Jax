import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers

class EigenNet(nn.Module):

    def setup(self):

        self.dense1 = nn.Dense(4)
        self.dense2 = nn.Dense(128)
        self.dense3 = nn.Dense(128)
        self.dense4 = nn.Dense(9)

    def __call__(self, x_in):
        x_in, D = x_in

        x = self.dense1(x_in)
        x = nn.softplus(x)
        x = self.dense2(x)
        x = nn.softplus(x)
        x = self.dense3(x)
        x = nn.softplus(x)
        x = self.dense4(x)

        d = jnp.sqrt(2*D**2 - x_in**2) - D
        d = jnp.prod(d, axis=-1, keepdims=True)
        x = x*d

        return x

    @staticmethod
    def sparsify_layer(W, sparsifing_K, l, L):
        m = W.shape[0]
        n = W.shape[1]

        x = np.linspace(0, m, m+1)
        y = np.linspace(0, n, n+1)
        xv, yv = np.meshgrid(x, y, indexing='ij')

        for k in range(sparsifing_K):
            alpha = k/(sparsifing_K - 1) * (l - 1)/L
            beta = (L-l+1)/L





if __name__ == '__main__':

    D = 2
    model = EigenNet()
    batch = jnp.ones((3, 2))
    variables = model.init(jax.random.PRNGKey(0), (batch, D))
    print(variables['params']['dense1']['kernel'])
    output = model.apply(variables, (batch, D))
    '''

    x = np.linspace(0, 2, 3)
    y = np.linspace(0, 2, 3)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    print(xv)
    print(yv)
    '''