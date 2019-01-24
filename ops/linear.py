import numpy as np
import tensorflow as tf

import pdb

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Linear(opts, scope, inputs_, output_dim, biases=True, weightnorm=None,
                                                        gain=1.,
                                                        reuse=None):
    """Fully connected linear layer.

    Args:
        input_: [num_points, ...] tensor, where every point can have an
            arbitrary shape. In case points are more than 1 dimensional,
            we will stretch them out in [numpoints, prod(dims)].
        output_dim: number of features for the output. I.e., the second
            dimensionality of the matrix W.
        initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """

    shpe = inputs_.get_shape().as_list()
    input_dim = np.prod(shpe[1:])

    with tf.variable_scope(scope or 'lin',reuse=reuse):

        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        if opts['mlp_init'] == 'lecun':# and input_dim != output_dim):
            # disabling orth. init for now because it's too slow
            weight_values = uniform(
                np.sqrt(1./input_dim),
                (input_dim, output_dim)
            )

        elif opts['mlp_init'] == 'glorot' or (opts['mlp_init'] == None):

            weight_values = uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif opts['mlp_init'] == 'he':

            weight_values = uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim)
            )

        elif opts['mlp_init'] == 'glorot_he':

            weight_values = uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif opts['mlp_init'] == 'orthogonal' or \
            (opts['mlp_init'] == None and input_dim == output_dim):

            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], np.prod(shape[1:]))
                 # TODO: why normal and not uniform?
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype('float32')
            weight_values = sample((input_dim, output_dim))

        elif opts['mlp_init'][0] == 'uniform':

            weight_values = np.random.uniform(
                low=-opts['mlp_init'][1],
                high=opts['mlp_init'][1],
                size=(input_dim, output_dim)
            ).astype('float32')

        else:

            raise Exception('Invalid initialization!')

        weight_values *= gain

        weight = tf.Variable(
            name=scope + '.W',
            initial_value=weight_values
        )

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # norm_values = np.linalg.norm(weight_values, axis=0)

            target_norms = tf.Variable(
                name=scope + '.g',
                initial_value=norm_values
            )

            with tf.name_scope('weightnorm') as scope_:
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        # if 'Discriminator' in name:
        #     print "WARNING weight constraint on {}".format(scope)
        #     weight = tf.nn.softsign(10.*weight)*.1

        if len(shpe) == 2:
            result = tf.matmul(inputs_, weight)
        else:
            reshaped_inputs = tf.reshape(inputs_, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            # result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs_))[:-1] + [output_dim]))

        if biases:
            result = tf.nn.bias_add(
                result,
                tf.Variable(
                    name=scope + '.b',
                    initial_value=np.zeros((output_dim,), dtype='float32')
                )
            )

        return result
