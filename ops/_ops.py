# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""Tensorflow ops used by GAN.

"""

import tensorflow as tf
import numpy as np
import logging

import pdb

def lrelu(x, leak=0.3):
    return tf.maximum(x, leak * x)

def upsample_nn(input_, new_size, scope=None, reuse=None):
    """NN up-sampling
    """

    with tf.variable_scope(scope or "upsample_nn", reuse=reuse):
        result = tf.image.resize_nearest_neighbor(input_, new_size)

    return result

def downsample(input_, d_h=2, d_w=2, conv_filters_dim=None, scope=None, reuse=None):
    """NN up-sampling
    """

    with tf.variable_scope(scope or "downsample", reuse=reuse):
        result = tf.nn.max_pool(input_, ksize=[1, d_h, d_w, 1], strides=[1, d_h, d_w, 1], padding='SAME')

    return result


def conv2d(opts, input_, output_dim, d_h=2, d_w=2, scope=None,
           conv_filters_dim=None, padding='SAME', l2_norm=False):
    """Convolutional layer.

    Args:
        input_: should be a 4d tensor with [num_points, dim1, dim2, dim3].

    """

    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d works only with 4d tensors.'

    with tf.variable_scope(scope or 'conv2d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))
        conv = tf.nn.bias_add(conv, biases)

    return conv

def deconv2d(opts, input_, output_shape, d_h=2, d_w=2, scope=None, conv_filters_dim=None, padding='SAME'):
    """Transposed convolution (fractional stride convolution) layer.

    """

    stddev = opts['init_std']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d_transpose works only with 4d tensors.'
    assert len(output_shape) == 4, 'outut_shape should be 4dimensional'

    with tf.variable_scope(scope or "deconv2d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

    return deconv

def logsumexp(logits,axis=1,keepdims=True):
    eps = 1e-06
    tmp = tf.reduce_sum(tf.exp(logits),axis=axis,keepdims=keepdims)
    return tf.log(tmp + eps)

def logsumexp_v2(logits, axis=1, keepdims=True):
    mean = tf.reduce_mean(logits, axis=axis, keepdims=keepdims)
    tmp = tf.reduce_sum(logits - mean, axis=axis, keepdims=keepdims)
    return tf.log(tmp) + mean

def softmax(logits,axis=None):
    return tf.nn.softmax(logits,axis=axis)

def non_linear(inputs,type):
    if type=='relu':
        return tf.nn.relu(inputs)
    elif type=='soft_plus':
        return tf.nn.softplus(inputs)
    elif type=='tanh':
        return tf.nn.tanh(inputs)
    elif type=='leaky_relu':
        alpha = .2
        return tf.maximum(alpha*inputs, inputs)
    else:
        assert False, 'Unknow non linear operation'
