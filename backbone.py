import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers

class EigenNet(nn.Module):


    def setup(self):

        self.dense1 = nn.Dense(128)
        self.dense2 = nn.Dense(128)
        self.dense3 = nn.Dense(128)
        self.dense4 = nn.Dense(9)

    def __call__(self, x):
        x, D = x

        x = self.dense1(x)
        x = nn.softplus(x)
        x = self.dense2(x)
        x = nn.softplus(x)
        x = self.dense3(x)
        x = nn.softplus(x)
        x = self.dense4(x)
        print(x)
        d = jnp.sqrt(2*D**2 - x**2) - D
        x = x*d
        print(x)

        #x = jnp.prod(jnp.sqrt(2*D**2 - x**2) - D)
        return x

if __name__ == '__main__':
    D = 1
    model = EigenNet()
    batch = jnp.ones((2, 1))
    variables = model.init(jax.random.PRNGKey(0), (batch,D))
    #output = model.apply(variables, batch)
    #print(output)