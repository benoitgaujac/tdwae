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
    # logSigma = tf.clip_by_value(logSigma, -20, 20)
    # Sigma = tf.exp(logSigma)
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
                                                        scope=None,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=0):
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
                                                        is_training)
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
                                                        is_training)
        elif archi == 'resnet':
            # Resnet archi similar to Imporved training of WAGAN
            outputs, out_shape = resnet_encoder(opts, input, 128, reuse,
                                                        is_training)

        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % archi)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)
    return mean, Sigma, out_shape

def mlp_encoder(opts, input, num_layers, num_units, output_dim,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=0):
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
        """
        layer_x = tf.nn.dropout(layer_x, rate=dropout_rate)
        """
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs

def dcgan_encoder(opts, input, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        reuse=False,
                                                        is_training=False):
    # Reshaping if needed
    shape = input.get_shape().as_list()
    if len(shape)<4:
        assert len(shape)==2, 'Wrong shape for inputs'
        # h_sqr = shape[-1]/datashapes[opts['dataset']][-1]
        h_sqr = shape[-1]
        w_sqr = h_sqr
        # reshape = (int(sqrt(h_sqr)),int(sqrt(w_sqr)),datashapes[opts['dataset']][-1])
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
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['e_nonlinearity'])
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs

def dcgan_v2_encoder(opts, input, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample=False,
                                                        reuse=False,
                                                        is_training=False):

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
                                                        is_training=False):
    output, output_shape = dcgan_v2_encoder(opts, input, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training)


def resnet_encoder(opts, input, output_dim, reuse=False, is_training=False):
    output = ops.resnet.OptimizedResBlockEnc1(opts,input,output_dim)
    output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res2', resample='down', reuse=reuse, is_training=is_training)
    output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res3', resample=None, reuse=reuse, is_training=is_training)
    output = ops.resnet.ResidualBlock(opts, output, output_dim, output_dim, 3, 'enc_res4', resample=None, reuse=reuse, is_training=is_training)
    output = ops._ops.non_linear(output,'relu')
    output = tf.reduce_mean(output, axis=[1,2])
    output = ops.linear.Linear(opts, output, np.prod(output.get_shape().as_list()[1:]), 2*output_dim, scope='hid_final')

    return output, None


def decoder(opts, input, archi, num_layers, num_units, filter_size,
                                                        output_dim,
                                                        features_dim=None,
                                                        resample=False,
                                                        scope=None,
                                                        reuse=False,
                                                        is_training=False,
                                                        dropout_rate=0.):
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
                                                        is_training)
        elif archi == 'dcgan_v2':
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = dcgan_v2_decoder(opts, input, archi, num_layers,
                                                        num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training)
        elif archi == 'resnet':
            # Resnet archi similar to Imporved training of WAGAN
            outputs = resnet_decoder(opts, input, output_dim, reuse,
                                                        is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['d_arch'])

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)

    return tf.layers.flatten(mean), tf.layers.flatten(Sigma)

def mlp_decoder(opts, input, num_layers, num_units, output_dim,
                                                        reuse,
                                                        is_training,
                                                        dropout_rate=0.):
    # Architecture with only fully connected layers and ReLUs
    layer_x = input
    for i in range(num_layers):
        layer_x = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    num_units, init=opts['mlp_init'], scope='hid%d/lin' % i)
        layer_x = ops._ops.non_linear(layer_x,opts['d_nonlinearity'])
        """
        layer_x = tf.nn.dropout(layer_x, rate=dropout_rate)
        """
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['d_norm']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        elif opts['d_norm']=='layernorm':
            layer_x = ops.layernorm.Layernorm(
                opts, layer_x, 'hid%d/bn' % i, reuse)
            # layer_x = ops.batchnorm.Batchnorm_contrib(
            #     opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
    outputs = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs

def  dcgan_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        reuse,
                                                        is_training):

    if output_dim==2*np.prod(datashapes[opts['dataset']]):
        h_sqr = output_dim / (2*datashapes[opts['dataset']][-1])
        w_sqr = h_sqr
        output_shape = (int(sqrt(h_sqr)),int(sqrt(w_sqr)),2*datashapes[opts['dataset']][-1])
    else:
        h_sqr = output_dim / 2
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
    _out_shape = [batch_size] + list(output_shape)
    if archi == 'dcgan':
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                    filter_size, scope='hid_final/deconv', init= opts['conv_init'])
    elif archi == 'dcgan_mod':
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                    filter_size, stride = 1, scope='hid_final/deconv', init= opts['conv_init'])

    return outputs

def  dcgan_v2_decoder(opts, input, archi, num_layers, num_units,
                                                        filter_size,
                                                        output_dim,
                                                        features_dim,
                                                        resample,
                                                        reuse,
                                                        is_training):
    # batch_size
    batch_size = tf.shape(input)[0]
    # Reshapping to linear
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
        # layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [-1,]+features_dim,
        #             filter_size, stride=1, scope='hid%d/deconv' % (i+1), init= opts['conv_init'])
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1], num_units,
                filter_size, stride=1, scope='hid%d/deconv' % (i+1), init=opts['conv_init'])
    # Final linear
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

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
