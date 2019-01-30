import numpy as np
import tensorflow as tf

import pdb

def custom_uniform(stdev, size):
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
    ).astype('float32')


def Conv2d(opts, input, input_dim, output_dim, filter_size, stride=1, padding='SAME', scope=None, init='he', biases=True):
    """Convolutional layer.

    Args:
        input: should be a 4d tensor with [num_points, dim1, dim2, dim3].

    """

    # shape = input.get_shape().as_list()
    # input_dim = shape[-1]
    if filter_size is None:
        filter_size = opts['filter_size']

    assert len(input.get_shape().as_list()) == 4, 'Conv2d works only with 4d tensors.'

    with tf.variable_scope(scope or 'conv2d'):
        if init=='he':
            fan_in = input_dim * filter_size**2
            fan_out = output_dim * filter_size**2 / (stride**2)
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
            filter_values = custom_uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim))
            w = tf.get_variable(
                'filter', initializer=filter_values)
        elif init=='normilized_glorot':
            fan_in = input_dim * filter_size**2
            fan_out = output_dim * filter_size**2 / (stride**2)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))
            filter_values = custom_uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim))
            w = tf.get_variable(
                'filter', initializer=filter_values)
        elif init=='truncated_norm':
            w = tf.get_variable(
                'filter', [filter_size, filter_size, shape[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=opts['init_std']))
        else:
            raise Exception('Invalid %s conv initialization!' % opts['conv_init'])
        conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)

        if biases:
            bias = tf.get_variable(
                'b', [output_dim],
                initializer=tf.constant_initializer(opts['init_bias']))
            conv = tf.nn.bias_add(conv, bias)

    return conv
