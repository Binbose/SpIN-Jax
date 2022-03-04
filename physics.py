import jax.numpy as jnp
import numpy as np
from helper import get_hessian_diagonals, get_hessian_diagonals_2
import time

def get_hydrogen_potential():

    def hygrogen_potential(x):
        return 1/jnp.linalg.norm(50 * x, axis=-1)

    return hygrogen_potential

def second_difference_along_coordinate(fn, fn_x, x , i, eps):
    coordinate = np.zeros_like(x)
    coordinate[:,i] = 1
    return fn(x + coordinate * eps) + fn(x - coordinate * eps) - 2 * fn_x

def hamiltonian_operator(fn, x, fn_x=None, nummerical_diff=True, eps=0.1, system='hydrogen'):
    if system == 'hydrogen':
        v_fn = get_hydrogen_potential()
    else:
        v_fn = get_hydrogen_potential()

    if fn_x is None:
        fn_x = fn(x)
    v = v_fn(x)[:, None] * fn_x
    if nummerical_diff:
        differences = 0
        for i in range(x.shape[1]):
            differences += second_difference_along_coordinate(fn, fn_x, x, i, eps)
        #second_derivative = differences
        second_derivative = differences / eps**2
    else:
        second_derivative = get_hessian_diagonals(fn, x)


    return -(second_derivative + v)
