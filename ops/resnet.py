import numpy as np
import tensorflow as tf

import functools

import pdb

from ops.linear import Linear
from ops.batchnorm import Batchnorm_layers
from ops.conv2d import Conv2d
import ops._ops


def ConvMeanPool(opts, input, input_dim, output_dim, filter_size, scope=None, init='he', biases=True):
    output = Conv2d(opts, input, input_dim, output_dim, filter_size, scope=scope, init=init, biases=biases)
    output = tf.add_n([output[:,::2,::2,:], output[:,1::2,::2,:], output[:,::2,1::2,:], output[:,1::2,1::2,:]]) / 4.
    return output

def MeanPoolConv(opts, input, input_dim, output_dim, filter_size, scope=None, init='he', biases=True):
    output = input
    output = tf.add_n([output[:,::2,::2,:], output[:,1::2,::2,:], output[:,::2,1::2,:], output[:,1::2,1::2,:]]) / 4.
    output = Conv2d(opts, output, input_dim, output_dim, filter_size, scope=scope, init=init, biases=biases)
    return output

def UpsampleConv(opts, input, input_dim, output_dim, filter_size, scope=None, init='he', biases=True):
    output = input
    output = tf.concat([output, output, output, output], axis=-1) # concat along channel axis
    # output = tf.concat([output, output, output, output], axis=1)
    # output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    # output = tf.transpose(output, [0,3,1,2])
    output = Conv2d(opts, output, input_dim, output_dim, filter_size, scope=scope, init=init, biases=biases)
    return output

def ResidualBlock(opts,input, input_dim, output_dim, filter_size, scope=None, resample=None, is_training=False, reuse=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(Conv2d, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(Conv2d, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = Conv2d
        conv_1        = functools.partial(Conv2d, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(Conv2d, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = input # Identity skip-connection
    else:
        shortcut = conv_shortcut(opts, input=input, input_dim=input_dim, output_dim=output_dim, filter_size=1, scope=scope+'.Shortcut',init='normilized_glorot', biases=True)

    output = input
    output = Batchnorm_layers(opts, output, scope=scope, is_training=is_training, reuse=reuse)
    output = ops._ops.non_linear(output,'relu')
    output = conv_1(opts,input=output,scope=scope+'/Conv1', filter_size=filter_size)
    output = Batchnorm_layers(opts, output, scope=scope+'/Bnv1', is_training=is_training, reuse=reuse)
    output = ops._ops.non_linear(output,'relu')
    output = conv_2(opts,input=output,scope=scope+'/Conv2', filter_size=filter_size)

    return shortcut + output

def OptimizedResBlockEnc1(opts, input, output_dim):
    conv_1        = functools.partial(Conv2d, input_dim=3, output_dim=output_dim)
    conv_2        = functools.partial(ConvMeanPool, input_dim=output_dim, output_dim=output_dim)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut(opts,input,input_dim=3, output_dim=output_dim, filter_size=1, scope='enc_res1/Shortcut', init='normilized_glorot', biases=True)
    output = input
    output = conv_1(opts,input=output,scope='enc_res1/Conv1', filter_size=3)
    output = ops._ops.non_linear(output,'relu')
    output = conv_2(opts,input=output,scope='enc_res1/Conv2', filter_size=3)
    return shortcut + output
