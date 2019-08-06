import numpy as np
import tensorflow as tf
from math import ceil, sqrt

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

def one_layer_encoder(opts, input, output_dim, norm, scope, reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        layer_x = input
        for i in range(len(opts['zdim'])):
            layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                        opts['e_nfilters'][i], init=opts['mlp_init'], scope='hid{}/lin_0'.format(i))
            if norm == 'batchnorm':
                layer_x = ops.batchnorm.Batchnorm_layers(
                    opts, layer_x, 'hid{}/bn_0'.format(i), is_training, reuse)
            elif norm == 'layernorm':
                layer_x = ops.layernorm.Layernorm(
                    opts, layer_x, 'hid{}/bn_0'.format(i), reuse)
            layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
            layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                        opts['e_nfilters'][i], init=opts['mlp_init'], scope='hid{}/lin_1'.format(i))
            if norm == 'batchnorm':
                layer_x = ops.batchnorm.Batchnorm_layers(
                    opts, layer_x, 'hid{}/bn_1'.format(i), is_training, reuse)
            elif norm == 'layernorm':
                layer_x = ops.layernorm.Layernorm(
                    opts, layer_x, 'hid{}/bn_1'.format(i), reuse)
            layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
            if i<len(opts['zdim'])-1:
                layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                            opts['zdim'][i], init=opts['mlp_init'], scope='hid{}/hid_final'.format(i))
        outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    output_dim, init=opts['mlp_init'], scope='hid_final')

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -50, 500)
    Sigma = tf.nn.softplus(logSigma)
    return mean, Sigma

def one_layer_decoder(opts, input, output_dim, norm, scope, reuse=False,
                                                        is_training=False):
    # Architecture with only fully connected layers and ReLUs
    with tf.variable_scope(scope, reuse=reuse):
        layer_x = input
        for i in range(len(opts['zdim'])-1,-1,-1):
            layer_x = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                        opts['d_nfilters'][i], init=opts['mlp_init'], scope='hid{}/lin_0'.format(i))
            layer_x = ops._ops.non_linear(layer_x,opts['d_nonlinearity'])
            if norm == 'batchnorm':
                layer_x = ops.batchnorm.Batchnorm_layers(
                    opts, layer_x, 'hid{}/bn_0'.format(i), is_training, reuse)
            elif norm == 'layernorm':
                layer_x = ops.layernorm.Layernorm(
                    opts, layer_x, 'hid{}/bn_0'.format(i), reuse)
            layer_x = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                        opts['d_nfilters'][i], init=opts['mlp_init'], scope='hid{}/lin_1'.format(i))
            layer_x = ops._ops.non_linear(layer_x,opts['d_nonlinearity'])
            if norm == 'batchnorm':
                layer_x = ops.batchnorm.Batchnorm_layers(
                    opts, layer_x, 'hid{}/bn_1'.format(i), is_training, reuse)
            elif norm == 'layernorm':
                layer_x = ops.layernorm.Layernorm(
                    opts, layer_x, 'hid{}/bn_1'.format(i), reuse)
            if i>0:
                layer_x = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                            opts['zdim'][i-1], init=opts['mlp_init'], scope='hid{}/hid_final'.format(i))
        outputs = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    output_dim, init=opts['mlp_init'], scope='hid_final')

    return tf.nn.sigmoid(outputs)


def encoder(opts, input, archi, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        features_dim=None,
                                                        resample=None,
                                                        last_archi=False,
                                                        top_latent=False,
                                                        scope=None,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=1.):
    with tf.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_encoder(opts, input, num_layers,
                                                        num_units,
                                                        output_dim,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
            out_shape = None
        elif archi == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_encoder(opts, input, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
            out_shape = None
        elif archi == 'dcgan_v2':
            # Fully convolutional architecture similar to Wasserstein GAN
            outputs, out_shape = dcgan_v2_encoder(opts, input, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
        elif archi == 'resnet':
            # Resnet archi similar to Imporved training of WAGAN
            outputs, out_shape = resnet_encoder(opts, input, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
        elif archi == 'resnet_v2':
            # Full conv Resnet archi similar to Imporved training of WAGAN
            outputs, out_shape = resnet_v2_encoder(opts, input, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        last_archi,
                                                        top_latent,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
        else:
            raise ValueError('%s : Unknown encoder architecture' % archi)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)
    return tf.layers.flatten(mean), tf.layers.flatten(Sigma), out_shape

def mlp_encoder(opts, input, num_layers, num_units, output_dim,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=1.):
    layer_x = input
    for i in range(num_layers):
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    num_units, init=opts['mlp_init'], scope='hid{}/lin'.format(i))
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['e_norm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['e_norm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs

def dcgan_encoder(opts, input, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=1.):
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
                filter_size, stride=2,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['e_norm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['e_norm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs

def dcgan_v2_encoder(opts, input, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample=False,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=1.):
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
                filter_size,stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['e_norm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['e_norm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    # Last conv resampling
    if resample=='down':
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    elif resample==None:
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],num_units,
                filter_size,stride=1,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    if opts['e_norm']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid%d/bn' % (i+1), is_training, reuse)
    elif opts['e_norm']=='layernorm':
        layer_x = ops.layernorm.Layernorm(
            opts, layer_x, 'hid%d/bn' % (i+1), reuse)
    layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
    layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    # Flaten layer
    out_shape = layer_x.get_shape().as_list()[1:]
    layer_x = tf.reshape(layer_x,[-1,np.prod(out_shape)])
    # Final linear
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs, out_shape

def resnet_encoder(opts, input, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample=False,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=1.):
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
                filter_size,stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['e_norm']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['e_norm']=='layernorm':
            conv = ops.layernorm.Layernorm(
                opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['e_nonlinearity'])
        conv = tf.nn.dropout(conv, keep_prob=dropout_rate)
    # Last conv resampling
    if resample=='down':
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    elif resample==None:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],num_units,
                filter_size,stride=1,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    out_shape = conv.get_shape().as_list()[1:]
    # -- Shortcut
    if resample=='down':
        shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid_shortcut',init=opts['conv_init'])
    elif resample==None:
        if conv.get_shape().as_list()[1:]==layer_x.get_shape().as_list()[1:]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],num_units,
                    filter_size,stride=1,scope='hid_shortcut',init=opts['conv_init'])
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    # -- Resnet output
    outputs = conv + shortcut
    if opts['e_norm']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
            opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['e_norm']=='layernorm':
        outputs = ops.layernorm.Layernorm(
            opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['e_nonlinearity'])
    outputs = tf.nn.dropout(outputs, keep_prob=dropout_rate)
    outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs, out_shape

def resnet_v2_encoder(opts, input, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample=False,
                                                        last_archi='conv1x1',
                                                        top_latent=False,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=1.):
    """
    Full conv resnet.
    output_dim:     number of output channels for intermediate latents / dimension of top latent
    features_dim:   shape of input [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to if needed features dim
    layer_x = tf.reshape(input,[-1,]+features_dim)
    # -- Conv block
    conv = layer_x
    for i in range(num_layers-1):
        conv = ops.conv2d.Conv2d(opts,conv,conv.get_shape().as_list()[-1],num_units,
                filter_size,stride=1,scope='hid{}/conv'.format(i),init=opts['conv_init'])
        if opts['e_norm']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['e_norm']=='layernorm':
            conv = ops.layernorm.Layernorm(
                opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['e_nonlinearity'])
        conv = tf.nn.dropout(conv, keep_prob=dropout_rate)
    # Last conv resampling
    if resample=='down':
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    elif resample==None:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1],num_units,
                filter_size,stride=1,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    # -- Shortcut
    if resample=='down':
        shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],2*num_units,
                filter_size,stride=2,scope='hid_shortcut',init='normilized_glorot')
    elif resample==None:
        if conv.get_shape().as_list()[1:]==layer_x.get_shape().as_list()[1:]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],num_units,
                    filter_size,stride=1,scope='hid_shortcut',init='normilized_glorot')
    else:
        assert False, 'Resample %s not allowed for encoder' % resample
    # -- Resnet output
    outputs = conv + shortcut
    if opts['e_norm']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
            opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['e_norm']=='layernorm':
        outputs = ops.layernorm.Layernorm(
            opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['e_nonlinearity'])
    outputs = tf.nn.dropout(outputs, keep_prob=dropout_rate)
    out_shape = outputs.get_shape().as_list()[1:-1] + [int(outputs.get_shape().as_list()[-1]),]
    # last hidden layer
    if last_archi=='dense':
        # -- dense layer
        outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                    output_dim, scope='hid_final')
    elif last_archi=='conv1x1':
        # -- 1x1 conv
        if top_latent:
            output_dim = int(output_dim / outputs.get_shape().as_list()[1] / outputs.get_shape().as_list()[2])
        outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1],output_dim,
                1,stride=1,scope='hid_final',init=opts['conv_init'])
        out_shape = outputs.get_shape().as_list()[1:-1] + [int(outputs.get_shape().as_list()[-1]/2),]
    # elif last_archi=='conv':
    #     # -- conv with "big kernel". output_shape: [output_dim,output_dim,1]
    #     out_shape = outputs.get_shape().as_list()[1:-1]
    #     W_size = out_shape[0] - int(output_dim/2) + 1
    #     outputs = ops.conv2d.Conv2d(opts,outputs,out_shape,2,
    #             W_size,stride=1,padding='VALID',scope='hid_final',init=opts['conv_init'])
    #     out_shape = outputs.get_shape().as_list()[1:-1] + [int(outputs.get_shape().as_list()[-1]/2),]
    else:
        assert False, 'Unknown last_archi %s ' % out_shape

    return outputs, out_shape


def decoder(opts, input, archi, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        features_dim=None,
                                                        resample=False,
                                                        last_archi='conv1x1',
                                                        scope=None,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=1.):
    with tf.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_decoder(opts, input, num_layers,
                                                        num_units,
                                                        output_dim,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
        elif archi == 'dcgan' or opts['d_arch'] == 'dcgan_mod':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
        elif archi == 'dcgan_v2':
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = dcgan_v2_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
        elif archi == 'resnet':
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = resnet_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
        elif archi == 'resnet_v2':
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = resnet_v2_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        last_archi,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['d_arch'])

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)

    return tf.layers.flatten(mean), tf.layers.flatten(Sigma)

def mlp_decoder(opts, input, num_layers, num_units, output_dim,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=1.):
    # Architecture with only fully connected layers and ReLUs
    layer_x = input
    for i in range(num_layers):
        layer_x = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    num_units, init=opts['mlp_init'], scope='hid%d/lin' % i)
        layer_x = ops._ops.non_linear(layer_x,opts['d_nonlinearity'])
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['d_norm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['d_norm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    outputs = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                np.prod(output_dim), init=opts['mlp_init'], scope='hid_final')

    return outputs

def  dcgan_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=1.):
    """
    DCGAN style network with stride 2 at each hidden deconvolution layers.
    First dense layer reshape to [out_h/2**num_layers,out_w/2**num_layers,num_units].
    Then num_layers deconvolutions with stride 2 and num_units filters.
    Last deconvolution output a 3-d latent code [out_h,out_w,2].
    """

    if np.prod(output_dim)==2*np.prod(datashapes[opts['dataset']]):
        h_sqr = output_dim / (2*datashapes[opts['dataset']][-1])
        w_sqr = h_sqr
        output_shape = (int(sqrt(h_sqr)),int(sqrt(w_sqr)),2*datashapes[opts['dataset']][-1])
    else:
        h_sqr = np.prod(output_dim) / 2
        w_sqr = h_sqr
        output_shape = (int(sqrt(h_sqr)),int(sqrt(w_sqr)),2)
    batch_size = tf.shape(input)[0]
    if archi == 'dcgan':
        height = output_shape[0] / 2**num_layers
        width = output_shape[1] / 2**num_layers
    elif archi == 'dcgan_mod':
        height = output_shape[0] / 2**(num_layers - 1)
        width = output_shape[1] / 2**(num_layers - 1)
    h0 = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            num_units * ceil(height) * ceil(width), scope='hid0/lin')
    if opts['d_norm']=='batchnorm':
        h0 = ops.batchnorm.Batchnorm_layers(
            opts, h0, 'hid0/bn_lin', is_training, reuse)
    elif opts['d_norm']=='layernorm':
        h0 = ops.layernorm.Layernorm(
            opts, h0, 'hid0/bn_lin', reuse)
    h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), num_units])
    h0 = ops._ops.non_linear(h0,opts['d_nonlinearity'])
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, ceil(height * scale),
                      ceil(width * scale), int(num_units / scale)]
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                   filter_size, scope='hid%d/deconv' % i, init= opts['conv_init'])
        if opts['d_norm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['d_norm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['d_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
    _out_shape = [batch_size] + list(output_shape)
    if archi == 'dcgan':
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                    filter_size, scope='hid_final/deconv', init= opts['conv_init'])
    elif archi == 'dcgan_mod':
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                    1, scope='hid_final/deconv', init= opts['conv_init'])

    return outputs

def  dcgan_v2_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=1.):
    """
    DCGAN style network. First deconvolution layer can have stride 2.
    First dense layer reshape to [features_dim[0]/2,features_dim[1]/2,2*num_units] if resampling up
    or [features_dim[0],features_dim[1],num_units] if no resampling.
    Then num_layers-1 deconvolutions with num_units filters.
    Final dense layer with output's dimension of output_dim.
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # Reshapping to linear
    if resample=='up':
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        # handeling padding
        if features_dim[0]%2==0:
            reshape = [int(features_dim[0]/2),int(features_dim[1]/2),2*num_units]
        else:
            reshape = [int((features_dim[0]+1)/2),int((features_dim[1]+1)/2),2*num_units]
    elif resample==None:
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        reshape = [features_dim[0], features_dim[1], num_units]
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    h0 = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            np.prod(reshape), scope='hid0/lin')
    h0 = tf.reshape(h0, [-1,]+ reshape)
    if opts['d_norm']=='batchnorm':
        h0 = ops.batchnorm.Batchnorm_layers(
                    opts, h0, 'hid0/bn', is_training, reuse)
    elif opts['d_norm']=='layernorm':
        h0 = ops.layernorm.Layernorm(
                    opts, h0, 'hid0/bn', reuse)
    h0 = ops._ops.non_linear(h0,opts['d_nonlinearity'])
    layer_x = h0
    # First deconv resampling
    if resample=='up':
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [batch_size,]+features_dim,
                    filter_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    elif resample==None:
        # layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [-1,]+features_dim,
        #             filter_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])

    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # Deconv block
    for i in range(num_layers - 1):
        if opts['d_norm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                        opts, layer_x, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['d_norm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                        opts, layer_x, 'hid%d/bn' % (i+1), reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['d_nonlinearity'])
        layer_x = tf.nn.dropout(layer_x, keep_prob=dropout_rate)
        # layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [-1,]+features_dim,
        #             filter_size, stride=1, scope='hid%d/deconv' % (i+1), init= opts['conv_init'])
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # Final linear
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                np.prod(output_dim), scope='hid_final')

    return outputs

def  resnet_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=1.):
    """
    Same than dcgan_v2 but with residual connection.
    Final hidden layer can be dense layer, 1x1 conv or big-kernel conv.
    output_dim:     shape/dim of output latent
    features_dim:   shape of ouput features [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to features dim
    if resample=='up':
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        if features_dim[0]%2==0:
            reshape = [int(features_dim[0]/2),int(features_dim[1]/2),2*num_units]
        else:
            reshape = [int((features_dim[0]+1)/2),int((features_dim[1]+1)/2),2*num_units]
    elif resample==None:
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        reshape = [features_dim[0], features_dim[1], num_units]
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    layer_x = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            np.prod(reshape), scope='hid0/lin')
    layer_x = tf.reshape(layer_x, [-1,]+ reshape)
    # -- Conv block
    conv = layer_x
    # First deconv resampling
    if resample=='up':
        conv = ops.deconv2d.Deconv2D(opts, conv, conv.get_shape().as_list()[-1], [batch_size,]+features_dim,
                    filter_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    elif resample==None:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # Deconv block
    for i in range(num_layers - 1):
        if opts['d_norm']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                        opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['d_norm']=='layernorm':
            conv = ops.layernorm.Layernorm(
                        opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['d_nonlinearity'])
        # conv = tf.nn.dropout(conv, keep_prob=dropout_rate)
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # -- Shortcut
    if resample=='up':
        shortcut = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [batch_size,]+features_dim,
                    filter_size, stride=2, scope='hid_shortcut', init=opts['conv_init'])
    elif resample==None:
        if conv.get_shape().as_list()[-1]==layer_x.get_shape().as_list()[-1]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1], num_units,
                        filter_size, stride=1, scope='hid_shortcut', init=opts['conv_init'])
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # -- Resnet output
    outputs = conv + shortcut
    if opts['d_norm']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
                    opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['d_norm']=='layernorm':
        outputs = ops.layernorm.Layernorm(
                    opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['d_nonlinearity'])
    outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                np.prod(output_dim), scope='hid_final')

    return outputs

def  resnet_v2_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        last_archi,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=1.):
    """
    Full conv resnet
    output_dim:     number of output channels
    features_dim:   shape of input latent [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to features dim
    if last_archi=='conv1x1' and np.prod(features_dim)==input.get_shape().as_list()[1:]:
        layer_x = tf.reshape(input,[-1,]+features_dim)
    else:
        layer_x = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),np.prod(features_dim), scope='hid_init')
        layer_x = tf.reshape(layer_x, [-1,] + features_dim)
    # -- Conv block
    conv = layer_x
    # First deconv resampling
    if resample=='up':
        output_shape = [batch_size,2*features_dim[0],2*features_dim[1],num_units]
        conv = ops.deconv2d.Deconv2D(opts, conv, conv.get_shape().as_list()[-1], output_shape,
                    filter_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    elif resample==None:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # Deconv block
    for i in range(num_layers - 1):
        if opts['d_norm']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                        opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['d_norm']=='layernorm':
            conv = ops.layernorm.Layernorm(
                        opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['d_nonlinearity'])
        # conv = tf.nn.dropout(conv, keep_prob=dropout_rate)
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # -- Shortcut
    if resample=='up':
        shortcut = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], output_shape,
                    filter_size, stride=2, scope='hid_shortcut', init='normilized_glorot')
    elif resample==None:
        if conv.get_shape().as_list()[-1]==layer_x.get_shape().as_list()[-1]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1], num_units,
                        filter_size, stride=1, scope='hid_shortcut', init='normilized_glorot')
    else:
        assert False, 'Resample {} not allowed for encoder'.format(resample)
    # -- Resnet output
    outputs = conv + shortcut
    if opts['d_norm']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
                    opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['d_norm']=='layernorm':
        outputs = ops.layernorm.Layernorm(
                    opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['d_nonlinearity'])
    """
    if last_archi=='dense':
        # -- dense
        ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),np.prod(output_dim), scope='hid_final')
    elif last_archi=='conv1x1':
        # -- 1x1 conv
        outputs = ops.conv2d.Conv2d(opts, outputs, outputs.get_shape().as_list()[-1], output_dim[-1],
                    1, stride=1, scope='hid_final', init=opts['conv_init'])
    elif last_archi=='conv':
        # -- conv with "big kernel". output_shape: [output_dim,output_dim,1]
        out_shape = outputs.get_shape().as_list()[1:-1]
        W_size = out_shape[0] - int(output_dim/2) + 1
        outputs = ops.conv2d.Conv2d(opts,outputs,out_shape,2,
                W_size,stride=1,padding='VALID',scope='hid_final',init=opts['conv_init'])
    else:
        assert False, 'Unknown last_archi %s ' % out_shape
    """
    # -- 1x1 conv
    outputs = ops.conv2d.Conv2d(opts, outputs, outputs.get_shape().as_list()[-1], output_dim[-1],
                1, stride=1, scope='hid_final', init=opts['conv_init'])

    return outputs
