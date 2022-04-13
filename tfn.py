import enum
from math import sqrt
import tensorflow as tf
import jax.numpy as jnp
import scipy.linalg
from flax import linen as nn
from typing import Any, Callable, Sequence, Optional
EPSILON = 1e-8
import jax

def get_eijk():
    """
    Constant Levi-Civita tensor

    Returns:
        tf.Tensor of shape [3, 3, 3]
    """
    eijk_ = jnp.zeros((3, 3, 3))
    eijk_ = eijk_.at[0, 1, 2].set(1.)
    eijk_ = eijk_.at[1, 2, 0].set(1.)
    eijk_ = eijk_.at[2, 0, 1].set(1.)
    eijk_ = eijk_.at[0, 2, 1].set(-1.)
    eijk_ = eijk_.at[2, 1, 0].set(-1.)
    eijk_ = eijk_.at[1, 0, 2].set(-1.)
    return eijk_


def norm_with_epsilon(input_tensor, axis=None, keepdims=False):
    """
    Regularized norm

    Args:
        input_tensor: tf.Tensor

    Returns:
        tf.Tensor normed over axis
    """
    return jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(input_tensor), axis=axis, keepdims=keepdims), EPSILON))


def ssp(x):
    """
    Shifted soft plus nonlinearity.

    Args:
        x: tf.Tensor

    Returns:
        tf.Tensor of same shape as x 
   """
    return jnp.logaddexp(x+jnp.log(.5), jnp.log(.5))


class rotation_equivariant_nonlinearity(nn.Module):
    nonlin: Callable = ssp
    biases_initializer: Callable = nn.initializers.zeros
    @nn.compact
    def __call__(self, x):
        """
        Rotation equivariant nonlinearity.

        The -1 axis is assumed to be M index (of which there are 2 L + 1 for given L).

        Args:
            x: tf.Tensor with channels as -2 axis and M as -1 axis.

        Returns:
            tf.Tensor of same shape as x with 3d rotation-equivariant nonlinearity applied.
        """
        shape = x.shape
        channels = shape[-2]
        representation_index = shape[-1]

        biases = self.param('bias', self.biases_initializer, channels)

        if representation_index == 1:
            return self.nonlin(x)
        else:
            norm = norm_with_epsilon(x, axis=-1)
            nonlin_out = self.nonlin(norm + biases)
            factor = nonlin_out / norm
            # Expand dims for representation index.
            return jnp.multiply(x, jnp.expand_dims(factor, axis=-1))


def difference_matrix(geometry):
    """
    Get relative vector matrix for array of shape [N, 3].

    Args:
        geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]

    Returns:
        Relative vector matrix with shape [N, N, 3]
    """
    # [N, 1, 3]
    ri = jnp.expand_dims(geometry, axis=1)
    # [1, N, 3]
    rj = jnp.expand_dims(geometry, axis=0)
    # [N, N, 3]
    rij = ri - rj
    return rij


def distance_matrix(geometry):
    """
    Get relative distance matrix for array of shape [N, 3].

    Args:
        geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]

    Returns:
        Relative distance matrix with shape [N, N]
    """
    # [N, N, 3]
    rij = difference_matrix(geometry)
    # [N, N]
    dij = norm_with_epsilon(rij, axis=-1)
    return dij


def random_rotation_matrix(key):
    """
    Generates a random 3D rotation matrix from axis and angle.

    Args:
        numpy_random_state: numpy random state object

    Returns:
        Random rotation matrix.
    """
    key1, key2 = jax.random.split(key)
    axis = jax.random.normal(key, (3,))
    axis /= jnp.linalg.norm(axis) + EPSILON
    theta = 2*jnp.pi*jax.random.uniform(key2)
    return rotation_matrix(axis, theta)


def rotation_matrix(axis, theta):
    return jax.scipy.linalg.expm(jnp.cross(jnp.eye(3), axis * theta))


# Layers for 3D rotation-equivariant network.

class R(nn.Module):
    nonlin: Callable = nn.softplus
    hidden_dim: Optional[int] = None
    output_dim: int = 1

    @nn.compact
    def __call__(self, inputs):
        input_dim = inputs.shape[-1]
        if self.hidden_dim is None:
            hidden_dim = input_dim
        
        x = nn.Dense(hidden_dim)(inputs)
        x = self.nonlin(x)
        x = nn.Dense(self.output_dim)(x)

        # [N, N, output_dim]
        return x


def unit_vectors(v, axis=-1):
    return v / norm_with_epsilon(v, axis=axis, keepdims=True)


def Y_2(rij):
    # rij : [N, N, 3]
    # x, y, z : [N, N]
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = jnp.maximum(jnp.sum(jnp.square(rij), axis=-1), EPSILON)
    # return : [N, N, 5]
    output = jnp.stack([x * y / r2,
                       y * z / r2,
                       (-jnp.square(x) - jnp.square(y) + 2. * jnp.square(z)) / (2 * jnp.sqrt(3) * r2),
                       z * x / r2,
                       (jnp.square(x) - jnp.square(y)) / (2. * r2)],
                      axis=-1)
    return output

class F_0(nn.Module):

    @nn.compact
    def __call__(self, inputs):
        # [N, N, output_dim, 1]
        return jnp.expand_dims(R()(inputs), axis=-1)

class F_1(nn.Module):

    @nn.compact
    def __call__(self, inputs, rij):
        # [N, N, output_dim]
        radial = R()(inputs)

        # Mask out for dij = 0
        dij = jnp.linalg.norm(rij, axis=-1)
        condition = jnp.tile(jnp.expand_dims(dij < EPSILON, axis=-1), [1, 1, self.output_dim])
        masked_radial = jnp.where(condition, jnp.zeros_like(radial), radial)

        # [N, N, output_dim, 3]
        return jnp.expand_dims(unit_vectors(rij), axis=-2) * jnp.expand_dims(masked_radial, axis=-1)

class F_2(nn.Module):

    @nn.compact
    def __call__(self, inputs, rij):
         # [N, N, output_dim]
        radial = R()(inputs)
        # Mask out for dij = 0
        dij = jnp.linalg.norm(rij, axis=-1)
        condition = jnp.tile(jnp.expand_dims(dij < EPSILON, axis=-1), [1, 1, self.output_dim])
        masked_radial = jnp.where(condition, jnp.zeros_like(radial), radial)
        # [N, N, output_dim, 5]
        return jnp.expand_dims(Y_2(rij), axis=-2) * jnp.expand_dims(masked_radial, axis=-1)

class filter_0(nn.Module):

    @nn.compact
    def __call__(self, layer_input, rbf_inputs):
         # [N, N, output_dim, 1]
        F_0_out = F_0()(rbf_inputs)
        # [N, output_dim]
        input_dim = layer_input.shape[-1]
        # Expand filter axis "j"
        cg = jnp.expand_dims(jnp.eye(input_dim), axis=-2)
        # L x 0 -> L
        return jnp.einsum('ijk,abfj,bfk->afi', cg, F_0_out, layer_input)

class filter_1_output_0(nn.Module):

    @nn.compact
    def __call__(self, layer_input, rbf_inputs, rij):
        # [N, N, output_dim, 3]
        F_1_out = F_1()(rbf_inputs, rij)
        # [N, output_dim, 3]
        if layer_input.shape[-1] == 1:
            raise ValueError("0 x 1 cannot yield 0")
        elif layer_input.shape[-1] == 3:
            # 1 x 1 -> 0
            cg = jnp.expand_dims(jnp.eye(3), axis=0)
            return jnp.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")


class filter_1_output_1(nn.Module):

    @nn.compact
    def __call__(self, layer_input, rbf_inputs, rij):
        # [N, N, output_dim, 3]
        F_1_out = F_1()(rbf_inputs, rij)
        # [N, output_dim, 3]
        if layer_input.shape[-1] == 1:
            # 0 x 1 -> 1
            cg = jnp.expand_dims(jnp.eye(3), axis=-1)
            return jnp.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
        elif layer_input.shape[-1] == 3:
            # 1 x 1 -> 1
            return jnp.einsum('ijk,abfj,bfk->afi', get_eijk(), F_1_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")

class filter_2_output_2(nn.Module):

    @nn.compact
    def __call__(self, layer_input, rbf_inputs, rij):
        # [N, N, output_dim, 3]
        F_2_out = F_2()(rbf_inputs, rij)
        # [N, output_dim, 5]
        if layer_input.shape[-1] == 1:
            # 0 x 2 -> 2
            cg = jnp.expand_dims(jnp.eye(5), axis=-1)
            return jnp.einsum('ijk,abfj,bfk->afi', cg, F_2_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")

# class self_interaction_layer(nn.Module):
#     output_dim: int
#     use_bias: bool
#     weights_initializer: Callable = nn.initializers.orthogonal()
#     biases_initializer: Callable = nn.initializers.zeros

#     @nn.compact
#     def __call__(self, inputs):
#         # input has shape [N, C, 2L+1]
#         # input_dim is number of channels
#         input_dim = inputs.shape[-2]
#         w_si = self.param('weights', self.weights_initializer, (self.output_dim, input_dim))

#         if self.use_bias:
#             b_si = self.param('biases', self.biases_initializer, (self.output_dim,))
#             return jnp.transpose(jnp.einsum('afi,gf->aig', inputs, w_si) + b_si, axes=[0, 2, 1])
#         else:
#             return jnp.transpose(jnp.einsum('afi,gf->aig', inputs, w_si), axes=[0, 2, 1])
#         # [N, output_dim, 2l+1]

class convolution(nn.Module):

    @nn.compact
    def __call__(self, input_tensor_list, rbf, unit_vectors):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                output_dim = tensor.shape[-2]
                if True:
                    # L x 0 -> L
                    tensor_out = filter_0(output_dim=output_dim)(tensor, rbf)
                    m = 0 if tensor_out.shape[-1] == 1 else 1
                    output_tensor_list[m].append(tensor_out)
                if key == 1:
                    # L x 1 -> 0
                    tensor_out = filter_1_output_0(output_dim=output_dim)(tensor, rbf, unit_vectors)
                    m = 0 if tensor_out.shape[-1] == 1 else 1
                    output_tensor_list[m].append(tensor_out)
                if key == 0 or key == 1:
                    # L x 1 -> 1
                    tensor_out = filter_1_output_1(output_dim=output_dim)(tensor, rbf, unit_vectors)
                    m = 0 if tensor_out.shape[-1] == 1 else 1
                    output_tensor_list[m].append(tensor_out)
        return output_tensor_list

class self_interaction(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                if key == 0:
                    tensor_out = nn.DenseGeneral(self.features[0], use_bias=True, axis=-2)(tensor)
                else:
                    tensor_out = nn.DenseGeneral(self.features[0], use_bias=False, axis=-2)(tensor)
                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
        return output_tensor_list

class nonlinearity(nn.Module):
    nonlin: Callable = nn.softplus

    @nn.compact
    def __call__(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                tensor_out = rotation_equivariant_nonlinearity(self.nonlin)(tensor)
                m = 0 if tensor_out.shape[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
        return output_tensor_list


def concatenation(input_tensor_list):
    output_tensor_list = {0: [], 1: []}
    for key in input_tensor_list:
        # Concatenate along channel axis
        # [N, channels, M]
        output_tensor_list[key].append(jnp.concatenate(input_tensor_list[key], axis=-2))
    return output_tensor_list


