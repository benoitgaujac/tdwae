import numpy as np
import tensorflow as tf
import ops
from datahandler import datashapes
from math import ceil, sqrt

import pdb


def encoder(opts, inputs, num_units, output_dim, scope, reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        if opts['e_arch'] == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_encoder(opts, inputs, opts['e_nlayers'],
                                                        num_units,
                                                        2*output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        elif opts['e_arch'] == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_encoder(opts, inputs, opts['e_nlayers'],
                                                        num_units,
                                                        2*output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['e_arch'])

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 20)
    Sigma = tf.nn.softplus(logSigma)
    return mean, Sigma

def mlp_encoder(opts, inputs, num_layers, num_units, output_dim,
                                                        batch_norm=False,
                                                        reuse=False,
                                                        is_training=False):
    layer_x = inputs
    for i in range(num_layers):
        scale = 2**i
        layer_x = ops.linear(opts, layer_x, int(num_units / scale),
                                            scope='hid{}/lin'.format(i))
        if batch_norm:
            layer_x = ops.batch_norm(opts, layer_x, is_training, reuse,
                                            scope='hid{}/bn'.format(i))
        layer_x = tf.nn.relu(layer_x)
    outputs = ops.linear(opts, layer_x, output_dim, scope='hid_final')

    return outputs

def dcgan_encoder(opts, inputs, num_layers, num_units, output_dim,
                                                        batch_norm=False,
                                                        reuse=False,
                                                        is_training=False):
    # Reshaping if needed
    shape = inputs.get_shape().as_list()
    if len(shape)<4:
        assert len(shape)==2, 'Wrong shape for inputs'
        reshape = (int(sqrt(shape[-1])),int(sqrt(shape[-1])),1)
        inputs = tf.reshape(inputs,(-1,)+reshape)
    layer_x = inputs
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = ops.conv2d(opts, layer_x, int(num_units / scale),
                                            scope='hid{}/conv'.format(i))
        if batch_norm:
            layer_x = ops.batch_norm(opts, layer_x, is_training, reuse,
                                            scope='hid{}/bn'.format(i))
        layer_x = tf.nn.relu(layer_x)
    outputs = ops.linear(opts, layer_x, output_dim, scope='hid_final')
    return outputs

def decoder(opts, inputs, num_units, output_shape, scope, reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        if opts['d_arch'] == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs, logits = mlp_decoder(opts, inputs, opts['d_nlayers'],
                                                        num_units,
                                                        output_shape,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        elif opts['d_arch'] == 'dcgan' or opts['d_arch'] == 'dcgan_mod':
            # Fully convolutional architecture similar to DCGAN
            outputs, logits = dcgan_decoder(opts, inputs, opts['d_arch'],
                                                        opts['d_nlayers'],
                                                        num_units,
                                                        output_shape,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['d_arch'])
    return outputs, logits


def mlp_decoder(opts, inputs, num_layers, num_units, outputs_shape,
                                                        batch_norm,
                                                        reuse,
                                                        is_training):
    # Architecture with only fully connected layers and ReLUs
    layer_x = inputs
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = ops.linear(opts, layer_x, int(num_units / scale),
                                                        scope='hid%d/lin' % i)
        layer_x = tf.nn.relu(layer_x)
        if batch_norm:
            layer_x = ops.batch_norm(
                opts, layer_x, is_training, reuse, scope='hid%d/bn' % i)
    out = ops.linear(opts, layer_x,
                     np.prod(outputs_shape), 'hid_final')
    out = tf.reshape(out, [-1] + list(outputs_shape))
    if opts['input_normalize_sym']:
        return tf.nn.tanh(out), out
    else:
        return tf.nn.sigmoid(out), out

def  dcgan_decoder(opts, inputs, archi, num_layers, num_units,
                                                        outputs_shape,
                                                        batch_norm,
                                                        reuse,
                                                        is_training):
    # Reshaping if needed
    if len(outputs_shape)<3:
        assert len(outputs_shape)==1, 'Wrong shape for inputs'
        output_shape = (int(sqrt(outputs_shape[0])),int(sqrt(outputs_shape[0])),1)
    else:
        output_shape = outputs_shape
    batch_size = tf.shape(inputs)[0]
    if archi == 'dcgan':
        height = output_shape[0] / 2**num_layers
        width = output_shape[1] / 2**num_layers
    elif archi == 'dcgan_mod':
        height = output_shape[0] / 2**(num_layers - 1)
        width = output_shape[1] / 2**(num_layers - 1)

    h0 = ops.linear(opts, inputs, num_units * ceil(height) * ceil(width),
                                                        scope='hid0/lin')
    h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, ceil(height * scale),
                      ceil(width * scale), int(num_units / scale)]
        layer_x = ops.deconv2d(opts, layer_x, _out_shape,
                               scope='hid%d/deconv' % i)
        if batch_norm:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='hid%d/bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)
    if archi == 'dcgan':
        last_h = ops.deconv2d(
            opts, layer_x, _out_shape, scope='hid_final/deconv')
    elif archi == 'dcgan_mod':
        last_h = ops.deconv2d(
            opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hid_final/deconv')
    last_h = tf.reshape(last_h,[-1]+list(outputs_shape))
    if opts['input_normalize_sym']:
        return tf.nn.tanh(last_h), last_h
    else:
        return tf.nn.sigmoid(last_h), last_h
