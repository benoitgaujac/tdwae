import numpy as np
import tensorflow as tf
from math import ceil, sqrt

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops._ops
from datahandler import datashapes

import pdb


def encoder(opts, inputs, archi, num_layers, num_units, output_dim, scope,
                                                        reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_encoder(opts, inputs, num_layers,
                                                        num_units,
                                                        2*output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        elif archi == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_encoder(opts, inputs, num_layers,
                                                        num_units,
                                                        2*output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % archi)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -50, 50)
    Sigma = tf.nn.softplus(logSigma)
    return mean, Sigma

def mlp_encoder(opts, inputs, num_layers, num_units, output_dim,
                                                        batch_norm=False,
                                                        reuse=False,
                                                        is_training=False):
    layer_x = inputs
    for i in range(num_layers):
        layer_x = ops.linear.Linear(opts=opts, input_=layer_x,
            output_dim=num_units, init=opts['mlp_init'], scope='hid{}/lin'.format(i))
        if batch_norm:
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse,)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #    opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['non_linearity'])
    outputs = ops.linear.Linear(opts=opts, input_=layer_x,
        output_dim=output_dim, init=opts['mlp_init'], scope='hid_final')

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
        # layer_x = ops._ops.conv2d(opts, layer_x, int(num_units / scale), opts['filter_size']
        #                                     scope='hid{}/conv'.format(i),init=opts['conv_init'])
        layer_x = ops.conv2d.Conv2d(opts, layer_x, int(num_units / scale),
                opts['filter_size'],scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if batch_norm:
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['non_linearity'])
    outputs = ops.linear.Linear(opts=opts,input_=layer_x,
        output_dim=output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs

def decoder(opts, inputs, archi, num_layers, num_units, output_dim, scope,
                                                        reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            mean, Sigma = mlp_decoder(opts, inputs, num_layers,
                                                        num_units,
                                                        2*output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        elif archi == 'dcgan' or opts['d_arch'] == 'dcgan_mod':
            # Fully convolutional architecture similar to DCGAN
            mean, Sigma = dcgan_decoder(opts, inputs, archi,
                                                        num_layers,
                                                        num_units,
                                                        2*output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['d_arch'])
    return mean, Sigma


def mlp_decoder(opts, inputs, num_layers, num_units, output_dim,
                                                        batch_norm,
                                                        reuse,
                                                        is_training):
    # Architecture with only fully connected layers and ReLUs
    layer_x = inputs
    for i in range(num_layers):
        layer_x = ops.linear.Linear(opts=opts, input_=layer_x,
            output_dim=num_units, init=opts['mlp_init'], scope='hid%d/lin' % i)
        layer_x = ops._ops.non_linear(layer_x,opts['non_linearity'])
        if batch_norm:
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
    out = ops.linear.Linear(opts=opts, input_=layer_x,
            output_dim=output_dim, init=opts['mlp_init'], scope='hid_final')
    mean, logSigma = tf.split(out,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -50, 50)
    Sigma = tf.nn.softplus(logSigma)

    return mean, Sigma


def  dcgan_decoder(opts, inputs, archi, num_layers, num_units,
                                                        output_dim,
                                                        batch_norm,
                                                        reuse,
                                                        is_training):

    if output_dim==2*np.prod(datashapes[opts['dataset']]):
        # Highest latent layer: reconstructions have data shape
        output_shape = datashapes[opts['dataset']]
        batch_size = tf.shape(inputs)[0]
        if archi == 'dcgan':
            height = output_shape[0] / 2**num_layers
            width = output_shape[1] / 2**num_layers
        elif archi == 'dcgan_mod':
            height = output_shape[0] / 2**(num_layers - 1)
            width = output_shape[1] / 2**(num_layers - 1)
        h0 = ops.linear.Linear(opts=opts, input_=inputs, output_dim=num_units * ceil(height) * ceil(width),
                                                            scope='hid0/lin')
        h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), num_units])
        h0 = ops._ops.non_linear(h0,opts['non_linearity'])
        layer_x = h0
        for i in range(num_layers - 1):
            scale = 2**(i + 1)
            _out_shape = [batch_size, ceil(height * scale),
                          ceil(width * scale), int(num_units / scale)]
            # layer_x = ops._ops.deconv2d(opts, layer_x, _out_shape,
            #            scope='hid%d/deconv' % i)
            layer_x = ops.deconv2d.Deconv2d(opts, layer_x, _out_shape,
                       scope='hid%d/deconv' % i, init= opts['conv_init'])
            if batch_norm:
                layer_x = ops.batchnorm.Batchnorm_layers(
                    opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
                # layer_x = ops.batchnorm.Batchnorm_contrib(
                #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
            layer_x = ops._ops.non_linear(layer_x,opts['non_linearity'])
        _out_shape = [batch_size] + list(output_shape)
        if archi == 'dcgan':
            # last_h = ops._ops.deconv2d(
            #             opts, layer_x, _out_shape, scope='hid_final/deconv')
            layer_h = ops.deconv2d.Deconv2d(opts, layer_x, _out_shape,
                        scope='hid_final/deconv' % i, init= opts['conv_init'])
        elif archi == 'dcgan_mod':
            # last_h = ops._ops.deconv2d(
            #             opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hid_final/deconv')
            layer_h = ops.deconv2d.Deconv2d(opts, layer_x, _out_shape,
                        stride = [1,1,1,1], scope='hid_final/deconv' % i, init= opts['conv_init'])
        return last_h, None
    else:
        # Deeper latent layers: reconstructions have shape (h,w,2)
        output_shape = (int(sqrt(output_dim/2)),int(sqrt(output_dim/2)),2)
        batch_size = tf.shape(inputs)[0]
        if archi == 'dcgan':
            height = output_shape[0] / 2**num_layers
            width = output_shape[1] / 2**num_layers
        elif archi == 'dcgan_mod':
            height = output_shape[0] / 2**(num_layers - 1)
            width = output_shape[1] / 2**(num_layers - 1)
        h0 = ops.linear.Linear(opts=opts,input_=inputs,
                output_dim=num_units * ceil(height) * ceil(width), scope='hid0/lin')
        h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), num_units])
        h0 = ops._ops.non_linear(h0,opts['non_linearity'])
        layer_x = h0
        for i in range(num_layers - 1):
            scale = 2**(i + 1)
            _out_shape = [batch_size, ceil(height * scale),
                          ceil(width * scale), int(num_units / scale)]
            # layer_x = ops._ops.deconv2d(opts, layer_x, _out_shape,
            #                        scope='hid%d/deconv' % i)
            layer_x = ops.deconv2d.Deconv2d(opts, layer_x, _out_shape,
                       scope='hid%d/deconv' % i, init= opts['conv_init'])
            if batch_norm:
                layer_x = ops.batchnorm.Batchnorm_layers(
                    opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
                # layer_x = ops.batchnorm.Batchnorm_contrib(
                #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
            layer_x = ops._ops.non_linear(layer_x,opts['non_linearity'])
        _out_shape = [batch_size] + list(output_shape)
        if archi == 'dcgan':
            # last_h = ops._ops.deconv2d(
            #     opts, layer_x, _out_shape, scope='hid_final/deconv')
            layer_h = ops.deconv2d.Deconv2d(opts, layer_x, _out_shape,
                        scope='hid_final/deconv' % i, init= opts['conv_init'])
        elif archi == 'dcgan_mod':
            # last_h = ops._ops.deconv2d(
            #     opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hid_final/deconv')
            layer_h = ops.deconv2d.Deconv2d(opts, layer_x, _out_shape,
                        stride = [1,1,1,1], scope='hid_final/deconv' % i, init= opts['conv_init'])
        mean, logSigma = tf.split(last_h,2,axis=-1)
        logSigma = tf.clip_by_value(logSigma, -50, 50)
        Sigma = tf.nn.softplus(logSigma)
        return tf.layers.flatten(mean), tf.layers.flatten(Sigma)
