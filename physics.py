import jax.numpy as jnp
from helper import get_hessian_diagonals, get_hessian_diagonals_2


def get_hydrogen_potential():
    def hygrogen_potential(x):
        return 1/jnp.linalg.norm(x, axis=-1)

    return hygrogen_potential


def hamiltonian_operator(fn, x, nummerical_diff=True, eps=0.1, system='hydrogen'):
    if system == 'hydrogen':
        v_fn = get_hydrogen_potential()

    v = v_fn(x)[:,None]
    if nummerical_diff:
        second_derivative = fn(x + eps) + fn(x - eps) - 2 * fn(x)
    else:

        second_derivative = get_hessian_diagonals(fn, x)


    return second_derivative + v
