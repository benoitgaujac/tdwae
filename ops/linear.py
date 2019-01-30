import numpy as np
import tensorflow as tf

import pdb


def custom_uniform(stdev, size):
    return np.random.uniform(low=-stdev * np.sqrt(3),
                            high=stdev * np.sqrt(3),
                            size=size
                            ).astype('float32')

def Linear(opts, input, input_dim, output_dim, scope=None, init=None, reuse=None):
    """Fully connected linear layer.

    Args:
        input: [num_points, ...] tensor, where every point can have an
            arbitrary shape. In case points are more than 1 dimensional,
            we will stretch them out in [numpoints, prod(dims)].
        output_dim: number of features for the output. I.e., the second
            dimensionality of the matrix W.
    """

    stddev = opts['init_std']
    bias_start = opts['init_bias']
    # input_dim = np.prod(shape[1:])

    shape = input.get_shape().as_list()
    assert len(shape) > 0
    if len(shape) > 2:
        # This means points contained in input have more than one
        # dimensions. In this case we first stretch them in one
        # dimensional vectors
        input = tf.reshape(input, [-1, input_dim])

    with tf.variable_scope(scope or "lin", reuse=reuse):
        if init == 'normal' or init == None:
            matrix = tf.get_variable(
                "W", [input_dim, output_dim], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        elif init == 'glorot':
            weight_values = custom_uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim))
            matrix = tf.get_variable(
                "W", initializer=weight_values, dtype=tf.float32)
        elif init == 'he':
            weight_values = custom_uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim))
            matrix = tf.get_variable(
                "W", initializer=weight_values, dtype=tf.float32)
        elif init == 'glorot_he':
            weight_values = custom_uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim))
            matrix = tf.get_variable(
                "W", initializer=weight_values, dtype=tf.float32)
        elif init == 'glorot_uniform':
            matrix = tf.get_variable(
                "W", [input_dim, output_dim], tf.float32,
                tf.glorot_uniform_initializer())
        elif init[0] == 'uniform':
            matrix = tf.get_variable(
                "W", [input_dim, output_dim], tf.float32,
                tf.random_uniform_initializer(
                    minval=-initialization[1],
                    maxval=initialization[1]))
        else:
            raise Exception('Invalid %s mlp initialization!' % opts['mlp_init'])

        bias = tf.get_variable(
            "b", [output_dim],
            initializer=tf.constant_initializer(bias_start))


    return tf.matmul(input, matrix) + bias
