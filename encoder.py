import numpy as np
import tensorflow as tf
from math import ceil, sqrt, log, exp

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops.layernorm
import ops._ops
import ops.resnet
from datahandler import datashapes

import logging
import pdb

def one_layer_encoder(opts, input, reuse=False, is_training=False):
    with tf.compat.v1.variable_scope('encoder', reuse=reuse):
        layer_x = input
        # -- looping over the latent layers
        for i in range(opts['nlatents']):
            with tf.compat.v1.variable_scope('layer_{}'.format(i+1), reuse=reuse):
                # -- looping over the hidden layers within latent layer i
                for j in range(opts['nlayers'][i]-1):
                    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                opts['e_nfilters'][i], init=opts['mlpinit'], scope='hid{}/lin'.format(j))
                    layer_x = ops.batchnorm.Batchnorm_layers(
                                opts, layer_x, 'hid{}/bn'.format(j), is_training, reuse)
                    layer_x = ops._ops.non_linear(layer_x,opts['nonlinearity'])
        # -- last hidden layer of latent layer
        with tf.compat.v1.variable_scope('layer_{}'.format(i+1), reuse=reuse):
            layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                        2*opts['zdim'][i], init=opts['mlpinit'], scope='final')

    mean, logSigma = tf.split(layer_x,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 20)
    Sigma = tf.nn.softplus(logSigma)
    return mean, Sigma

def Encoder(opts, input, archi, nlayers, nfilters, filters_size,
                                            output_dim=None,
                                            # features_dim=None,
                                            downsample=None,
                                            output_layer='mlp',
                                            scope=None,
                                            reuse=False,
                                            is_training=True):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_encoder(opts, input, nlayers,
                                            nfilters,
                                            output_dim,
                                            reuse,
                                            is_training)
        elif archi == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_encoder(opts, input, num_layers,
                                            num_units,
                                            filters_size,
                                            output_dim,
                                            reuse,
                                            is_training)
        elif archi == 'dcgan_v2':
            # Fully convolutional architecture similar to Wasserstein GAN
            outputs, out_shape = dcgan_v2_encoder(opts, input, num_layers,
                                            num_units,
                                            filters_size,
                                            output_dim,
                                            features_dim,
                                            downsample,
                                            reuse,
                                            is_training)
        elif archi == 'resnet':
            # Resnet archi similar to Imporved training of WAGAN
            outputs = resnet_encoder(opts, input, num_layers,
                                            num_units,
                                            filters_size,
                                            output_dim,
                                            features_dim,
                                            downsample,
                                            reuse,
                                            is_training)
        elif archi == 'resnet_v2':
            # Full conv Resnet archi similar to Imporved training of WAGAN
            outputs = resnet_v2_encoder(opts, input, num_layers,
                                            num_units,
                                            filters_size,
                                            output_dim,
                                            features_dim,
                                            downsample,
                                            last_archi,
                                            top_latent,
                                            reuse,
                                            is_training)
        else:
            raise ValueError('%s : Unknown encoder architecture' % archi)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    min, max = log(exp(1e-10)-1), 1e4
    logSigma = tf.clip_by_value(logSigma, min, max)
    Sigma = tf.nn.softplus(logSigma)
    return tf.compat.v1.layers.flatten(mean), tf.compat.v1.layers.flatten(Sigma)

def mlp_encoder(opts, input, nlayers, nunits, output_dim, reuse=False,
                                            is_training=False):
    layer_x = input
    for i in range(nlayers):
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    nunits, init=opts['mlpinit'], scope='hid{}/lin'.format(i))
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['enorm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['enorm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['nonlinearity'])
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                2*output_dim, init=opts['mlpinit'], scope='hid_final')

    return outputs

def dcgan_encoder(opts, input, num_layers, num_units, filters_size,
                                                        output_dim,
                                                        reuse=False,
                                                        is_training=False):
    """
    DCGAN style network with stride 2 at each hidden convolution layers.
    Final dense layer with output of size output_dim.
    """

    # Reshaping if needed
    shape = input.get_shape().as_list()
    if len(shape)<4:
        assert len(shape)==2, 'Wrong shape for inputs'
        h_sqr = shape[-1]
        w_sqr = h_sqr
        reshape = (int(sqrt(h_sqr)),int(sqrt(w_sqr)),1)
        input = tf.reshape(input,(-1,)+reshape)
    layer_x = input
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1], int(num_units / scale),
                filters_size, stride=2,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['enorm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['enorm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['nonlinearity'])
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                2*output_dim, scope='hid_final')

    return outputs

def dcgan_v2_encoder(opts, input, num_layers, num_units, filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        downsample=False,
                                                        reuse=False,
                                                        is_training=False):
    """
    DCGAN style network. Latent dimensions are not square, so need for a dense layer to reshape.
    First dense layer reshape to features_dim.
    Then num_layers-1 convolution layer with stride 1. Last convolution layer reample if needed.
    Final dense layer with output of size output_dim.
    output_dim:     dim of output latent
    features_dim:   shape of input FEATURES [w,h,c]
    """


    layer_x = input
    # Reshapping to features_dim if needed
    if layer_x.get_shape().as_list()[1:-1]!=features_dim[:-1]:
        layer_x = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    np.prod(features_dim), scope='hid_linear_init')
        layer_x = tf.reshape(layer_x,[-1,]+features_dim)
    # Conv block
    for i in range(num_layers-1):
        layer_x = ops.conv2d.Conv2d(opts,layer_x,layer_x.get_shape().as_list()[-1],num_units,
                filters_size,stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['enorm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['enorm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['nonlinearity'])
    # Last conv resampling
    if downsample:
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*num_units,
                filters_size,stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    else:
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],num_units,
                filters_size,stride=1,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    if opts['enorm']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid%d/bn' % (i+1), is_training, reuse)
    elif opts['enorm']=='layernorm':
        layer_x = ops.layernorm.Layernorm(
            opts, layer_x, 'hid%d/bn' % (i+1), reuse)
    layer_x = ops._ops.non_linear(layer_x,opts['nonlinearity'])
    # Flaten layer
    out_shape = layer_x.get_shape().as_list()[1:]
    layer_x = tf.reshape(layer_x,[-1,np.prod(out_shape)])
    # Final linear
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                2*output_dim, scope='hid_final')

    return outputs, out_shape

def resnet_encoder(opts, input, num_layers, num_units, filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        downsample=False,
                                                        reuse=False,
                                                        is_training=False):
    """
    Same than dcgan_v2 but with residual connection.
    output_dim:     dim of output latent
    features_dim:   shape of input FEATURES [w,h,c]
    """

    layer_x = input
    # -- Reshapping to features_dim if needed
    if layer_x.get_shape().as_list()[1:-1]!=features_dim[:-1]:
        layer_x = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    np.prod(features_dim), scope='hid_linear_init')
        layer_x = tf.reshape(layer_x,[-1,]+features_dim)
    # -- Conv block
    conv = layer_x
    for i in range(num_layers-1):
        conv = ops.conv2d.Conv2d(opts,conv,conv.get_shape().as_list()[-1],num_units,
                filters_size,stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['enorm']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['enorm']=='layernorm':
            conv = ops.layernorm.Layernorm(
                opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['nonlinearity'])
    # Last conv resampling
    if downsample:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],2*num_units,
                filters_size,stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    else:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],num_units,
                filters_size,stride=1,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    out_shape = conv.get_shape().as_list()[1:]
    # -- Shortcut
    if downsample:
        shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*num_units,
                filters_size,stride=2,scope='hid_shortcut',init=opts['conv_init'])
    else:
        if conv.get_shape().as_list()[1:]==layer_x.get_shape().as_list()[1:]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],num_units,
                    filters_size,stride=1,scope='hid_shortcut',init=opts['conv_init'])
    # -- Resnet output
    outputs = conv + shortcut
    if opts['enorm']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
            opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['enorm']=='layernorm':
        outputs = ops.layernorm.Layernorm(
            opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['nonlinearity'])
    outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                2*output_dim, scope='hid_final')

    return outputs, out_shape

def resnet_v2_encoder(opts, input, num_layers, num_units, filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        downsample=False,
                                                        last_archi='conv1x1',
                                                        top_latent=False,
                                                        reuse=False,
                                                        is_training=False):
    """
    Full conv resnet.
    output_dim:     number of output channels for intermediate latents / dimension of top latent
    features_dim:   shape of input [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to if needed features dim
    if input.get_shape().as_list()[1:]!=features_dim:
        layer_x = tf.reshape(input,[-1,]+features_dim)
    else:
        layer_x = input
    # -- Conv block
    conv = layer_x
    for i in range(num_layers-1):
        conv = ops.conv2d.Conv2d(opts,conv,conv.get_shape().as_list()[-1],num_units,
                filters_size,stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['enorm']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['enorm']=='layernorm':
            conv = ops.layernorm.Layernorm(
                opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['nonlinearity'])
    # Last conv resampling
    if downsample:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],2*num_units,
                filters_size,stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    else:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],num_units,
                filters_size,stride=1,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    # -- Shortcut
    if downsample:
        shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*num_units,
                filters_size,stride=2,scope='hid_shortcut',init='normilized_glorot')
    else:
        if conv.get_shape().as_list()[1:]==layer_x.get_shape().as_list()[1:]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],num_units,
                    filters_size,stride=1,scope='hid_shortcut',init='normilized_glorot')
    # -- Resnet output
    outputs = conv + shortcut
    if opts['enorm']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
            opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['enorm']=='layernorm':
        outputs = ops.layernorm.Layernorm(
            opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['nonlinearity'])

    # Shape
    if top_latent:
        output_dim = int(2*output_dim / outputs.get_shape().as_list()[1] / outputs.get_shape().as_list()[2])
    else:
        output_dim = 2*output_dim
    out_shape = outputs.get_shape().as_list()[1:-1] + [int(output_dim),]
    # last hidden layer
    if last_archi=='dense':
        # -- dense layer
        outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                    np.prod(out_shape), scope='hid_final')
    elif last_archi=='conv1x1':
        # -- 1x1 conv
        outputs = ops.conv2d.Conv2d(opts,outputs,outputs.get_shape().as_list()[-1],output_dim,
                1,stride=1,scope='hid_final',init=opts['conv_init'])
    elif last_archi=='conv':
        # -- conv
        outputs = ops.conv2d.Conv2d(opts,outputs,outputs.get_shape().as_list()[-1],output_dim,
                filters_size,stride=1,scope='hid_final',init=opts['conv_init'])
    else:
        assert False, 'Unknown last_archi %s ' % last_archi

    return outputs, out_shape
