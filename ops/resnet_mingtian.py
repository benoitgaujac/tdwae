import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

def int_shape(x):
    return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars


def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


@add_arg_scope
def dense(x_, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=[int(x_.get_shape()[1]), num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x_, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # x = tf.identity(x)
                x = tf.matmul(x_, V)
                scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
                x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x



@add_arg_scope
def conv2d(x_, num_filters, filter_size=[3, 3], stride=[2, 2, 2, 2], pad='SAME', nonlinearity=None, init_scale=1.,
           counters={}, init=False, ema=None, **kwargs):
    ''' convolutional layer '''
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size + [int(x_.get_shape()[-1]), num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x_, W, stride, pad), b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # x = tf.identity(x)
                W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
                x = tf.nn.bias_add(tf.nn.conv2d(x_, W, [1] + stride + [1], pad), b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x



@add_arg_scope
def deconv2d(x_, num_filters, filter_size=[3, 3], stride=[2, 2, 2, 2], pad='SAME', nonlinearity=None, init_scale=1.,
             counters={}, init=False, ema=None, **kwargs):
    ''' transposed convolutional layer '''
    name = get_name('deconv2d', counters)
    xs = int_shape(x_)
    if pad == 'SAME':
        target_shape = [xs[0], xs[1] * stride[0], xs[2] * stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1] * stride[0] + filter_size[0] - 1, xs[2] * stride[1] + filter_size[1] - 1,
                        num_filters]
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size + [num_filters, int(x_.get_shape()[-1])], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(x_, W, target_shape, stride , padding=pad)
        x = tf.nn.bias_add(x, b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # x = tf.identity(x)
                W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])
                x = tf.nn.conv2d_transpose(x_, W, target_shape, [1] + stride + [1], padding=pad)
                x = tf.nn.bias_add(x, b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x




@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])




@add_arg_scope
def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=False, counters={}, ema=None,
                 dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_filters)
    if a is not None:  # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
    c2 = conv(c1, num_filters * 2, init_scale=0.1)

    # add projection of h vector if included: conditional generation
    if h is not None:
        with tf.variable_scope(get_name('conditional_weights', counters)):
            hw = get_var_maybe_avg('hw', ema, shape=[int_shape(h)[-1], 2 * num_filters], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        if init:
            hw = hw.initialized_value()
        c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])

    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)
    return x + c3
