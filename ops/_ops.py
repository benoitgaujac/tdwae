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
