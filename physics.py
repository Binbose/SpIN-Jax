import jax.numpy as jnp
import numpy as np
from helper import get_hessian_diagonals
import time
import jax
from functools import partial
from jax import jit, vmap
from helper import compute_hessian_diagonals, vectorized_diagonal, vectorized_trace

def get_hydrogen_potential():

    def hygrogen_potential(x):
        return - 1/jnp.linalg.norm(x, axis=-1)

    return hygrogen_potential

def second_difference_along_coordinate(weight_dict, fn, x , i, eps):
    coordinate = np.zeros_like(x)
    coordinate[:,i] = 1
    return fn(weight_dict, x + coordinate * eps) + fn(weight_dict, x - coordinate * eps) - 2 * fn(weight_dict, x)


def laplace_numerical(fn, eps=0.1):
    def _laplace_numerical(weight_dict, x):
        differences = 0
        
        for i in range(2):
            differences += second_difference_along_coordinate(weight_dict, fn, x, i, eps)
        laplacian = differences / eps ** 2

        return laplacian

    return _laplace_numerical

def construct_hamiltonian_function(fn, system='hydrogen', eps=0.0):
    def _construct(weight_dict, x, fn_x):
        if eps==0.0:
            vectorized_hessian_result = vectorized_hessian(weight_dict, x)
            batch, n_eigenfunc, c1, c2 = vectorized_hessian_result.shape[0], vectorized_hessian_result.shape[1], \
                                         vectorized_hessian_result.shape[2], vectorized_hessian_result.shape[3]
            vectorized_hessian_result = vectorized_hessian_result.reshape(batch * n_eigenfunc, c1, c2)
            laplace = vectorized_trace(vectorized_hessian_result).reshape(batch, n_eigenfunc, -1)[:,:,0]
        else:
            laplace = vectorized_hessian(weight_dict, x)

        return -laplace + v_fn(x)[:,None] * fn_x
        # return v_fn(x)[:,None] * fn_x

    if system == 'hydrogen':
        v_fn = get_hydrogen_potential()
    elif system == 'laplace':
        v_fn = lambda x: 0*x.sum(-1)
    else:
        print('System "{}" not supported'.format(system))
        exit()

    if eps > 0.0:
        vectorized_hessian = laplace_numerical(fn, eps=eps)
    else:
        hessian = jax.hessian(fn, argnums=1)
        vectorized_hessian = vmap(hessian, in_axes=[None, 0])

    return _construct

