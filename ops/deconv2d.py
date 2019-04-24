import numpy as np
import tensorflow as tf

import pdb

def custom_uniform(stdev, size):
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
    ).astype('float32')


def Deconv2D(opts, input, input_dim, output_shape, filter_size=3, stride=2, scope=None, init='he', padding='SAME', biases=True):
    """2D Transposed convolution (fractional stride convolution) layer.
    input: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, height, width, output_dim)
    """

    # shape = input.get_shape().as_list()
    # input_dim = shape[-1]
    output_dim = output_shape[-1]
    if filter_size is None:
        filter_size = opts['filter_size']

    with tf.variable_scope(scope or "deconv2d"):
        if init=='he':
            fan_in = input_dim * filter_size / stride
            fan_out = output_dim * filter_size
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
            filter_values = custom_uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim))
            w = tf.get_variable(
                'filter', initializer=filter_values)
        elif init=='normilized_glorot':
            fan_in = input_dim * filter_size / stride
            fan_out = output_dim * filter_size
            filters_stdev = np.sqrt(2./(fan_in+fan_out))
            filter_values = custom_uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim))
            w = tf.get_variable(
                'filter', initializer=filter_values)
        elif init=='truncated_norm':
            w = tf.get_variable(
                'filter', [filter_size, filter_size, output_dim, input_dim],
                initializer=tf.random_normal_initializer(stddev=opts['init_std']))
        else:
            raise Exception('Invalid %s conv initialization!' % opts['conv_init'])
        deconv = tf.nn.conv2d_transpose(
            input, w, output_shape=output_shape,
            strides=[1, stride, stride, 1], padding=padding)

        if biases:
            biais = tf.get_variable(
                'b', [output_dim],
                initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, biais)

    return deconv
