import numpy as np
import tensorflow as tf

def Batchnorm_contrib(opts, input, scope=None, is_training=False, reuse=None, scale=True, center=True, fused=False):
    """Batch normalization based on tf.contrib.layers.

    """
    return tf.contrib.layers.batch_norm(
        input, center=center, scale=scale,
        epsilon=opts['batch_norm_eps'], decay=opts['batch_norm_momentum'],
        is_training=is_training, reuse=reuse, updates_collections=None,
        scope=scope, fused=fused)

def Batchnorm_layers(opts, input, scope=None, is_training=False, reuse=None, scale=True, center=True, fused=False):
    """Batch normalization based on tf.compat.v1.layers.batch_normalization.

    """
    return tf.compat.v1.layers.batch_normalization(
        input, center=center, scale=scale,
        epsilon=opts['batch_norm_eps'], momentum=opts['batch_norm_momentum'],
        training=is_training, reuse=reuse,
        name=scope, fused=fused)
