import numpy as np
import tensorflow as tf


def Layernorm(opts, input, scope=None, reuse=None, scale=True, center=True):
    """Layer normalization based on tf.contrib.layers.layer_norm.

    """
    return tf.contrib.layers.layer_norm(input, center=center, scale=scale, reuse=reuse, scope=scope)
