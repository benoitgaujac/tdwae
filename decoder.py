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

def one_layer_decoder(opts, input, reuse=False, is_training=False):
    with tf.compat.v1.variable_scope('decoder', reuse=reuse):
        layer_x = input
        # -- looping over the latent layers
        for i in range(opts['nlatents']-1,-1,-1):
            with tf.compat.v1.variable_scope('layer_{}'.format(i+1), reuse=reuse):
                # -- looping over the hidden layers within latent layer i
                for j in range(opts['nlayers'][i]):
                    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                opts['d_nfilters'][i], init=opts['mlpinit'], scope='hid{}/lin'.format(j))
                    layer_x = ops.batchnorm.Batchnorm_layers(
                        opts, layer_x, 'hid{}/bn'.format(j), is_training, reuse)
                    layer_x = ops._ops.non_linear(layer_x,opts['nonlinearity'])
        # -- last hidden layer of latent layer
        with tf.compat.v1.variable_scope('layer_{}'.format(i+1), reuse=reuse):
            layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                        np.prod(datashapes[opts['dataset']]), init=opts['mlpinit'], scope='final')

    return tf.nn.sigmoid(layer_x)


def Decoder(opts, input, archi, nlayers, nfilters, filters_size,
                                            output_dim=None,
                                            # features_dim=None,
                                            upsample=False,
                                            output_layer='mlp',
                                            scope=None,
                                            reuse=False,
                                            is_training=True):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_decoder(opts, input, nlayers,
                                            nfilters,
                                            output_dim,
                                            reuse,
                                            is_training)
        elif archi == 'dcgan' or opts['arch'] == 'dcgan_mod':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filters_size,
                                                        output_dim,
                                                        reuse,
                                                        is_training)
        elif archi == 'dcgan_v2':
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = dcgan_v2_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        upsample,
                                                        reuse,
                                                        is_training)
        elif archi == 'resnet':
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = resnet_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        upsample,
                                                        reuse,
                                                        is_training)
        elif archi == 'resnet_v2':
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = resnet_v2_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        upsample,
                                                        last_archi,
                                                        reuse,
                                                        is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['arch'])

    if output_dim is not None:
        mean, logSigma = tf.split(outputs,2,axis=-1)
        min, max = log(exp(1e-10)-1), 1e4
        logSigma = tf.clip_by_value(logSigma, min, max)
        Sigma = tf.nn.softplus(logSigma)
        mean = tf.compat.v1.layers.flatten(mean)
        Sigma = tf.compat.v1.layers.flatten(Sigma)
    else:
        mean, Sigma = tf.compat.v1.layers.flatten(outputs), None

    return mean, Sigma

def mlp_decoder(opts, input, nlayers, nunits, output_dim, reuse,
                                            is_training):
    # Architecture with only fully connected layers and ReLUs
    layer_x = input
    for i in range(nlayers):
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    nunits, init=opts['mlpinit'], scope='hid%d/lin' % i)
        layer_x = ops._ops.non_linear(layer_x, opts['nonlinearity'])
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['dnorm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['dnorm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
        layer_x = ops._ops.non_linear(layer_x, opts['nonlinearity'])
    if output_dim is not None:
        outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    2*np.prod(output_dim), init=opts['mlpinit'], scope='hid_final')
    else:
        outputs = layer_x

    return outputs

def  dcgan_decoder(opts, input, archi, num_layers, num_units,
                                                        filters_size,
                                                        output_dim,
                                                        reuse,
                                                        is_training):
    """
    DCGAN style network with stride 2 at each hidden deconvolution layers.
    First dense layer reshape to [out_h/2**num_layers,out_w/2**num_layers,num_units].
    Then num_layers deconvolutions with stride 2 and num_units filters.
    Last deconvolution output a 3-d latent code [out_h,out_w,2].
    """

    if np.prod(output_dim)==np.prod(datashapes[opts['dataset']]):
        h_sqr = output_dim / datashapes[opts['dataset']][-1]
        w_sqr = h_sqr
        output_shape = (int(sqrt(h_sqr)),int(sqrt(w_sqr)),2*datashapes[opts['dataset']][-1])
    else:
        h_sqr = np.prod(output_dim)
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
    if opts['dnorm']=='batchnorm':
        h0 = ops.batchnorm.Batchnorm_layers(
            opts, h0, 'hid0/bn_lin', is_training, reuse)
    elif opts['dnorm']=='layernorm':
        h0 = ops.layernorm.Layernorm(
            opts, h0, 'hid0/bn_lin', reuse)
    h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), num_units])
    h0 = ops._ops.non_linear(h0,opts['nonlinearity'])
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, ceil(height * scale),
                      ceil(width * scale), int(num_units / scale)]
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                   filters_size, scope='hid%d/deconv' % i, init= opts['conv_init'])
        if opts['dnorm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['dnorm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['nonlinearity'])
    _out_shape = [batch_size] + list(output_shape)
    if archi == 'dcgan':
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                    filters_size, scope='hid_final/deconv', init= opts['conv_init'])
    elif archi == 'dcgan_mod':
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                    1, scope='hid_final/deconv', init= opts['conv_init'])

    return outputs

def  dcgan_v2_decoder(opts, input, archi, num_layers, num_units,
                                                        filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        upsample,
                                                        reuse,
                                                        is_training):
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
    if upsample:
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        # handeling padding
        if features_dim[0]%2==0:
            reshape = [int(features_dim[0]/2),int(features_dim[1]/2),2*num_units]
        else:
            reshape = [int((features_dim[0]+1)/2),int((features_dim[1]+1)/2),2*num_units]
    else:
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        reshape = [features_dim[0], features_dim[1], num_units]
    h0 = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            np.prod(reshape), scope='hid0/lin')
    h0 = tf.reshape(h0, [-1,]+ reshape)
    if opts['dnorm']=='batchnorm':
        h0 = ops.batchnorm.Batchnorm_layers(
                    opts, h0, 'hid0/bn', is_training, reuse)
    elif opts['dnorm']=='layernorm':
        h0 = ops.layernorm.Layernorm(
                    opts, h0, 'hid0/bn', reuse)
    h0 = ops._ops.non_linear(h0,opts['nonlinearity'])
    layer_x = h0
    # First deconv resampling
    if upsample:
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [batch_size,]+features_dim,
                    filters_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    else:
        # layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [-1,]+features_dim,
        #             filters_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1], num_units,
                filters_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])

    # Deconv block
    for i in range(num_layers - 1):
        if opts['dnorm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                        opts, layer_x, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['dnorm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                        opts, layer_x, 'hid%d/bn' % (i+1), reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['nonlinearity'])
        # layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [-1,]+features_dim,
        #             filters_size, stride=1, scope='hid%d/deconv' % (i+1), init= opts['conv_init'])
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1], num_units,
                filters_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # Final linear
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                2*np.prod(output_dim), scope='hid_final')

    return outputs

def  resnet_decoder(opts, input, archi, num_layers, num_units,
                                                        filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        upsample,
                                                        reuse,
                                                        is_training):
    """
    Same than dcgan_v2 but with residual connection.
    Final hidden layer can be dense layer, 1x1 conv or big-kernel conv.
    output_dim:     shape/dim of output latent
    features_dim:   shape of ouput features [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to features dim
    if upsample:
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        if features_dim[0]%2==0:
            reshape = [int(features_dim[0]/2),int(features_dim[1]/2),2*num_units]
        else:
            reshape = [int((features_dim[0]+1)/2),int((features_dim[1]+1)/2),2*num_units]
    else:
        if num_units!=features_dim[2]:
            logging.error('num units decoder not matching num_units decoder')
        reshape = [features_dim[0], features_dim[1], num_units]
    layer_x = ops.linear.Linear(opts,input,np.prod(input.get_shape().as_list()[1:]),
            np.prod(reshape), scope='hid0/lin')
    layer_x = tf.reshape(layer_x, [-1,]+ reshape)
    # -- Conv block
    conv = layer_x
    # First deconv resampling
    if upsample:
        output_shape = [batch_size,features_dim[0],features_dim[1],num_units]
        conv = ops.deconv2d.Deconv2D(opts, conv, conv.get_shape().as_list()[-1], output_shape,
                    filters_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    else:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filters_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
    # Deconv block
    for i in range(num_layers - 1):
        if opts['dnorm']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                        opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['dnorm']=='layernorm':
            conv = ops.layernorm.Layernorm(
                        opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['nonlinearity'])
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filters_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # -- Shortcut
    if upsample:
        output_shape = [batch_size,features_dim[0],features_dim[1],num_units]
        shortcut = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], output_shape,
                    filters_size, stride=2, scope='hid_shortcut', init=opts['conv_init'])
    else:
        if conv.get_shape().as_list()[-1]==layer_x.get_shape().as_list()[-1]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1], num_units,
                        filters_size, stride=1, scope='hid_shortcut', init=opts['conv_init'])
    # -- Resnet output
    outputs = conv + shortcut
    if opts['dnorm']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
                    opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['dnorm']=='layernorm':
        outputs = ops.layernorm.Layernorm(
                    opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['nonlinearity'])
    if np.prod(output_dim)==np.prod(features_dim):
        outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1], 2*output_dim[-1],
                filters_size, stride=1, scope='hid_final', init=opts['conv_init'])
    else:
        outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                    2*np.prod(output_dim), scope='hid_final')

    return outputs

def  resnet_v2_decoder(opts, input, archi, num_layers, num_units,
                                                        filters_size,
                                                        output_dim,
                                                        features_dim,
                                                        upsample,
                                                        last_archi,
                                                        reuse,
                                                        is_training):
    """
    Full conv resnet
    output_dim:     number of output channels
    features_dim:   shape of input latent [w,h,c]
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # -- Reshapping to features dim
    layer_x = tf.reshape(input,[-1,]+features_dim)

    conv = layer_x
    # First deconv resampling
    if upsample:
        output_shape = [batch_size,2*features_dim[0],2*features_dim[1],int(num_units/2)]
        conv = ops.deconv2d.Deconv2D(opts, conv, conv.get_shape().as_list()[-1], output_shape,
                    filters_size, stride=2, scope='hid0/deconv', init=opts['conv_init'])
    else:
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filters_size, stride=1, scope='hid0/deconv', init=opts['conv_init'])
    # Deconv block
    for i in range(num_layers - 1):
        if opts['dnorm']=='batchnorm':
            conv = ops.batchnorm.Batchnorm_layers(
                        opts, conv, 'hid%d/bn' % (i+1), is_training, reuse)
        elif opts['dnorm']=='layernorm':
            conv = ops.layernorm.Layernorm(
                        opts, conv, 'hid%d/bn' % (i+1), reuse)
        conv = ops._ops.non_linear(conv,opts['nonlinearity'])
        conv = ops.conv2d.Conv2d(opts, conv,conv.get_shape().as_list()[-1], num_units,
                filters_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # -- Shortcut
    if upsample:
        output_shape = [batch_size,2*features_dim[0],2*features_dim[1],num_units]
        shortcut = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], output_shape,
                    filters_size, stride=2, scope='hid_shortcut', init='normilized_glorot')
    else:
        if conv.get_shape().as_list()[-1]==layer_x.get_shape().as_list()[-1]:
            shortcut = layer_x
        else:
            shortcut = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1], num_units,
                        filters_size, stride=1, scope='hid_shortcut', init='normilized_glorot')
    # -- Resnet output
    outputs = conv + shortcut
    if opts['dnorm']=='batchnorm':
        outputs = ops.batchnorm.Batchnorm_layers(
                    opts, outputs, 'hid%d/bn' % (i+2), is_training, reuse)
    elif opts['dnorm']=='layernorm':
        outputs = ops.layernorm.Layernorm(
                    opts, outputs, 'hid%d/bn' % (i+2), reuse)
    outputs = ops._ops.non_linear(outputs,opts['nonlinearity'])

    # last hidden layer
    if last_archi=='dense':
        # -- dense layer
        if np.prod(output_dim)==np.prod(datashapes[opts['dataset']]):
            outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1],2*output_dim[-1],
                    filters_size,stride=1,scope='hid_final',init=opts['conv_init'])
        else:
            output_shape = [outputs.get_shape().as_list()[1],outputs.get_shape().as_list()[2],2*output_dim[-1]]
            outputs = ops.linear.Linear(opts,outputs,np.prod(outputs.get_shape().as_list()[1:]),
                        np.prod(output_shape), scope='hid_final')
    elif last_archi=='conv1x1':
        # -- 1x1 conv
        outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1],2*output_dim[-1],
                1,stride=1,scope='hid_final',init=opts['conv_init'])
    elif last_archi=='conv':
        # -- conv
        outputs = ops.conv2d.Conv2d(opts, outputs,outputs.get_shape().as_list()[-1],2*output_dim[-1],
                filters_size,stride=1,scope='hid_final',init=opts['conv_init'])
    else:
        assert False, 'Unknown last_archi %s ' % last_archi


    return outputs
