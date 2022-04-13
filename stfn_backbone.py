from cmath import exp
import jax.numpy as jnp 



import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
# Useful dataclass to keep train weight_dict
from flax.training import train_state
from flax.core import FrozenDict
from matplotlib.pyplot import axis, xkcd

import numpy as np                     # Ordinary NumPy
from typing import Sequence
from flax.linen import jit
from jax import random, jit, vmap, jacfwd
from jax.nn import initializers
from jax import dtypes

class STFN_Net(nn.Module):
    D_min: float  # Dimension of the [-D,D]^2 box where we sample input x_in.
    D_max: float
    mask_type = 'quadratic'
    n_neuron: Sequence[int]
    n_eigenfuncs: int
    @nn.compact
    def __call__(self, x_in):
        if type(x_in) == tuple:
            x_in, L_inv = x_in
        else:
            L_inv = None

        x = x_in
        use_bias=True
        activation = jax.nn.softplus # softplus works better than signoid
        initilization = initializers.lecun_normal

        x_norm = jnp.linalg.norm(x,axis=-1)[...,jnp.newaxis]
        normalized_x= x_in / x_norm

        coeff = nn.Dense(self.n_neuron[0], use_bias=use_bias, kernel_init=initilization())(x_norm)
        coeff = activation(coeff)

        for i, feat in enumerate(self.n_neuron[1:]):
            coeff = nn.Dense(feat, use_bias=use_bias, kernel_init=initilization())(coeff)
            coeff = activation(coeff)

        n_coeff = 4
        coeff = nn.Dense(self.n_eigenfuncs*n_coeff, use_bias=use_bias, kernel_init=initilization())(coeff)
        coeff = coeff.reshape([-1,n_coeff]) #best: *0.02 or *10 make it bilevel opt

        tiled_x = jnp.tile(normalized_x,(1,self.n_eigenfuncs))
        tiled_x = tiled_x.reshape([-1,3])
        f = tiled_x*coeff[...,1:]
        
        f = jnp.sum((f),axis=-1).reshape([-1,self.n_eigenfuncs])
        f = f + coeff[...,0].reshape([-1,self.n_eigenfuncs])
        

        if self.mask_type == 'quadratic':
            # We multiply the output by \prod_i (\sqrt{2D^2-x_i^2}-D) to apply a boundary condition \psi(D_max) = 0 and \psi(D_min) = 0
            # See page 16th for more information
            D_avg = (self.D_max + self.D_min) / 2
            lim = self.D_max - D_avg
            d = (jnp.sqrt(2 * lim ** 2 - (x_in - D_avg) ** 2) - lim) / lim
            d = jnp.prod(d, axis=-1, keepdims=True) 
            f = f * d
        elif self.mask_type == 'exp':
            # Mask with gaussian instead to satisfy boundary condition \psi(x) -> 0 for x -> \infty
            # Standard deviation of gaussian is learnable
            mean = (self.D_max + self.D_min) / 2
            sigma = jnp.max(jnp.array([self.D_max, self.D_min])) / 4
            # embedding = jnp.abs(nn.Embed(1, self.features[-1], embedding_init=constant(sigma))(jnp.eye(1, dtype=‘int32’)))
            # sigma = (embedding * jnp.eye(k))[0]
            # print(embedding)
            normalization = 1 / (jnp.sqrt(2 * jnp.pi) * sigma)
            d = normalization * jnp.exp(-0.5 * jnp.linalg.norm(x_in - mean, axis=-1, keepdims=True) ** 2 / sigma ** 2)
            f = f * d

        if L_inv is not None:
            f = jnp.einsum('ij, bj -> bi', L_inv, f)

        return f

def get_angles(xyz):
    xy = xyz[0]**2 + xyz[1]**2
    ptsnew = jnp.array([jnp.arctan2(jnp.sqrt(xy), xyz[2]),jnp.arctan2(xyz[1], xyz[0])])
    return ptsnew
"""
def get_spherical(angle,coeff):
    phi, theta = angle
    c0,c1,c2,c3,shift = coeff
    r = jnp.sin(phi)
    return c0+c1*r*jnp.cos(theta+shift)+c2*r*jnp.cos(theta)+c3*jnp.cos(phi)
"""
def get_output(x,coeff):
    c0,c1,c2,c3 = coeff
    return c0+c1*x[0]+c2*x[1]+c3*x[2]

def get_spherical(angle,coeff):
    phi, theta = angle
    c0,c1,c2,c3 = coeff
    r = jnp.sin(phi)
    return c0+c1*r*jnp.sin(theta)+c2*r*jnp.cos(theta)+c3*jnp.cos(phi)

ONE_VECTOR = jnp.array([1.0,0.0,0.0])

def rotation_matrix_to_one(vec):
    """ Find the rotation matrix that aligns vec to [1,0,0]
    Modified from
    https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

    """
    a, b = (vec / jnp.linalg.norm(vec)).reshape(3), (ONE_VECTOR / jnp.linalg.norm(ONE_VECTOR)).reshape(3)
    v = jnp.cross(a, b)
    c = jnp.dot(a, b)
    s = jnp.linalg.norm(v)
    kmat = jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = jnp.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotation_matrix_from_one(vec):
    a, b = (ONE_VECTOR / jnp.linalg.norm(ONE_VECTOR)).reshape(3), (vec / jnp.linalg.norm(vec)).reshape(3)
    v = jnp.cross(a, b)
    c = jnp.dot(a, b)
    s = jnp.linalg.norm(v)
    kmat = jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = jnp.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix