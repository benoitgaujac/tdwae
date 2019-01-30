import numpy as np
import tensorflow as tf
from math import ceil, sqrt

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops._ops
import ops.resnet
from datahandler import datashapes

import pdb


def encoder(opts, input, archi, num_layers, num_units, output_dim, scope,
                                                        reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_encoder(opts, input, num_layers,
                                                        num_units,
                                                        output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        elif archi == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_encoder(opts, input, num_layers,
                                                        num_units,
                                                        output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        elif archi == 'resnet':
            # Resnet archi similar to Imporved training of WAGAN
            outputs = resnet_encoder(opts, input, 128, reuse,
                                                        is_training)

        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % archi)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -50, 50)
    Sigma = tf.nn.softplus(logSigma)
    return mean, Sigma

def mlp_encoder(opts, input, num_layers, num_units, output_dim,
                                                        batch_norm=False,
                                                        reuse=False,
                                                        is_training=False):
    layer_x = input
    for i in range(num_layers):
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    num_units, init=opts['mlp_init'], scope='hid{}/lin'.format(i))
        if batch_norm:
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse,)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #    opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs

def dcgan_encoder(opts, input, num_layers, num_units, output_dim,
                                                        batch_norm=False,
                                                        reuse=False,
                                                        is_training=False):
    # Reshaping if needed
    shape = input.get_shape().as_list()
    if len(shape)<4:
        assert len(shape)==2, 'Wrong shape for inputs'
        h_sqr = shape[-1]/datashapes[opts['dataset']][-1]
        w_sqr = h_sqr
        reshape = (int(sqrt(h_sqr)),int(sqrt(w_sqr)),datashapes[opts['dataset']][-1])
        input = tf.reshape(input,(-1,)+reshape)
    layer_x = input
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1], int(num_units / scale),
                opts['filter_size'],stride=2,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if batch_norm:
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs

def resnet_encoder(opts, input, output_dim, reuse=False, is_training=False):
    output = ops.resnet.OptimizedResBlockEnc1(opts,input,output_dim)
    output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res2', resample='down', reuse=reuse, is_training=is_training)
    output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res3', resample=None, reuse=reuse, is_training=is_training)
    output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res4', resample=None, reuse=reuse, is_training=is_training)
    output = ops._ops.non_linear(output,'relu')
    output = tf.reduce_mean(output, axis=[1,2])
    output = ops.linear.Linear(opts, output, np.prod(output.get_shape().as_list()[1:]), 2*output_dim, scope='hid_final')

    return output


def decoder(opts, input, archi, num_layers, num_units, output_dim, scope,
                                                        reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_decoder(opts, input, num_layers,
                                                        num_units,
                                                        output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        elif archi == 'dcgan' or opts['d_arch'] == 'dcgan_mod':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_decoder(opts, input, archi,
                                                        num_layers,
                                                        num_units,
                                                        output_dim,
                                                        opts['batch_norm'],
                                                        reuse,
                                                        is_training)
        elif archi == 'resnet':
            # Resnet archi similar to Imporved training of WAGAN
            outputs = resnet_decoder(opts, input, output_dim, reuse,
                                                        is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['d_arch'])

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -50, 50)
    Sigma = tf.nn.softplus(logSigma)

    return tf.layers.flatten(mean), tf.layers.flatten(Sigma)

def mlp_decoder(opts, input, num_layers, num_units, output_dim,
                                                        batch_norm,
                                                        reuse,
                                                        is_training):
    # Architecture with only fully connected layers and ReLUs
    layer_x = input
    for i in range(num_layers):
        layer_x = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    num_units, init=opts['mlp_init'], scope='hid%d/lin' % i)
        layer_x = ops._ops.non_linear(layer_x,opts['d_nonlinearity'])
        if batch_norm:
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
    outputs = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs

def  dcgan_decoder(opts, input, archi, num_layers, num_units,
                                                        output_dim,
                                                        batch_norm,
                                                        reuse,
                                                        is_training):

    h_sqr = output_dim / (2*datashapes[opts['dataset']][-1])
    w_sqr = h_sqr
    output_shape = (int(sqrt(h_sqr)),int(sqrt(w_sqr)),2*datashapes[opts['dataset']][-1])
    batch_size = tf.shape(input)[0]
    if archi == 'dcgan':
        height = output_shape[0] / 2**num_layers
        width = output_shape[1] / 2**num_layers
    elif archi == 'dcgan_mod':
        height = output_shape[0] / 2**(num_layers - 1)
        width = output_shape[1] / 2**(num_layers - 1)
    h0 = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            num_units * ceil(height) * ceil(width), scope='hid0/lin')
    if batch_norm:
        h0 = ops.batchnorm.Batchnorm_layers(
            opts, h0, 'hid0/bn_lin', is_training, reuse)
    h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), num_units])
    h0 = ops._ops.non_linear(h0,opts['d_nonlinearity'])
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, ceil(height * scale),
                      ceil(width * scale), int(num_units / scale)]
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                   scope='hid%d/deconv' % i, init= opts['conv_init'])
        if batch_norm:
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['d_nonlinearity'])
    _out_shape = [batch_size] + list(output_shape)
    if archi == 'dcgan':
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                    scope='hid_final/deconv', init= opts['conv_init'])
    elif archi == 'dcgan_mod':
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                    stride = [1,1,1,1], scope='hid_final/deconv', init= opts['conv_init'])

    return outputs

def resnet_decoder(opts, input, output_dim, reuse=False, is_training=False):
    input_dim = np.prod(input.get_shape().as_list()[1:])
    output = ops.linear.Linear(opts,input,input_dim,4*4*input_dim,scope='hid0/lin')
    output = tf.reshape(output, [-1, 4, 4, input_dim])
    output = ops.resnet.ResidualBlock(opts, output, input_dim, input_dim, 3, 'dec_res1', resample='up', reuse=reuse, is_training=is_training)
    output = ops.resnet.ResidualBlock(opts, output, input_dim, input_dim, 3, 'dec_res2', resample='up', reuse=reuse, is_training=is_training)
    output = ops.resnet.ResidualBlock(opts, output, input_dim, input_dim, 3, 'dec_res3', resample='up', reuse=reuse, is_training=is_training)
    output = ops.batchnorm.Batchnorm_layers(opts, output, 'hid3/bn', is_training, reuse)
    output = ops._ops.non_linear(output,'relu')
    output = ops.conv2d.Conv2d(opts, output, input_dim, 2*3, 3, scope='hid_final/conv', init='normilized_glorot')
    return output
