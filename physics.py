import jax.numpy as jnp
import numpy as np
from helper import get_hessian_diagonals, get_hessian_diagonals_2
import time
import jax

def get_hydrogen_potential():

    def hygrogen_potential(x):
        return - 1/jnp.linalg.norm(x, axis=-1)

    return hygrogen_potential

def second_difference_along_coordinate(fn, fn_x, x , i, eps):
    coordinate = np.zeros_like(x)
    coordinate[:,i] = 1
    return fn(x + coordinate * eps) + fn(x - coordinate * eps) - 2 * fn_x


def hamiltonian_operator(fn, x, fn_x=None, nummerical_diff=True, eps=0.1, system='hydrogen'):
    if system == 'hydrogen':
        v_fn = get_hydrogen_potential()
    elif system == 'laplace':
        v_fn = lambda x: 0*x
    else:
        v_fn = lambda x: 0*x


    if fn_x is None:
        fn_x = fn(x)
    v_fn = v_fn(x)
    if len(v_fn.shape) != len(fn_x.shape):
        v_fn = v_fn[:, None]

    v = v_fn * fn_x
    if nummerical_diff:
        differences = 0
        for i in range(x.shape[1]):
            differences += second_difference_along_coordinate(fn, fn_x, x, i, eps)
        laplacian = differences / eps**2

    else:
        laplacian = get_hessian_diagonals(fn, x).sum(-1)


    return laplacian + v
