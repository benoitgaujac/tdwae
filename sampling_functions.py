import sys
import time
import os
from math import sqrt, cos, sin, pi, ceil
import numpy as np
import tensorflow as tf

# import tensorflow_probability as tfp

import pdb


def sample_pz(opts, pz_params, batch_size=100):
    if opts["prior"] == "gaussian" or opts["prior"] == "implicit":
        noise = sample_gaussian(opts, pz_params, "numpy", batch_size)
    elif opts["prior"] == "dirichlet":
        noise = sample_dirichlet(opts, pz_params, batch_size)
    else:
        assert False, "Unknown prior %s" % opts["prior"]
    return noise


def sample_gaussian(opts, params, typ="numpy", batch_size=100):
    """
    Sample noise from gaussian distribution with parameters
    means and covs
    """
    if typ == "tensorflow":
        means, covs = tf.split(params, 2, axis=-1)
        shape = tf.shape(means)
        eps = tf.compat.v1.random_normal(shape, dtype=tf.float32)
        noise = means + tf.multiply(eps, tf.sqrt(1e-6 + covs))
    elif typ == "numpy":
        means, covs = np.split(params, 2, axis=-1)
        # means = means[:,0]
        # covs = covs[:,0]
        shape = (batch_size,) + np.shape(means)
        eps = np.random.normal(0.0, 1.0, shape).astype(np.float32)
        noise = means + np.multiply(eps, np.sqrt(1e-6 + covs))
    return noise


def sample_dirichlet(opts, alpha, batch_size=100):
    """
    Sample noise from dirichlet distribution with parameters
    alpha
    """
    return np.random.dirichlet(alpha, batch_size)


def sample_unif(shape, minval=0, maxval=None, dtype=tf.float32):
    """
    Sample noise from Unif[minval,maxval]
    """
    return tf.random.uniform(shape, minval, maxval, dtype)


def sample_bernoulli(params):
    """
    Sample noise from Bernoulli distribution with mean parameters
    params
    """
    # assert False, 'tfp not available on cluster gpu yet'
    bernoulli_dist = tf.compat.v1.distributions.Bernoulli(probs=params)
    return bernoulli_dist.sample()

    # eps = tf.random_uniform(shape)
    # return tf.where(tf.random_uniform(shape) - params < 0,
    #                         tf.ones(shape), tf.zeros(shape))


def linespace(opts, n, anchors):
    """
    Genereate linear grid space
        - n:  Num of steps in the interpolation
        - anchors[num_interpolations,2,zdim]: start/end of the linear interpolations
    Return:
    linespce[num_interpolations,num_steps,zdim]: list of linear interpolations
    """
    linespce = []
    num_inter = np.shape(anchors)[0]
    for i in range(num_inter):
        x = np.linspace(anchors[i, 0], anchors[i, 1], n, endpoint=True)
        linespce.append(x)
    linespce = np.stack(linespce, axis=0)

    return linespce


def generate_linespace(opts, n, mode, anchors):
    """
    Genereate latent linear grid space
    """
    nanchors = np.shape(anchors)[0]
    dim_to_interpolate = min(opts["nmixtures"], opts["zdim"])
    if mode == "transformation":
        assert np.shape(anchors)[1] == 0, "Zdim needs to be 2 to plot transformation"
        ymin, xmin = np.amin(anchors, axis=0)
        ymax, xmax = np.amax(anchors, axis=0)
        x = np.linspace(1.1 * xmin, 1.1 * xmax, n)
        y = np.linspace(1.1 * ymin, 1.1 * ymax, n)
        linespce = np.stack(np.meshgrid(y, x)).T
    elif mode == "points_interpolation":
        assert np.shape(anchors)[0] % 2 == 0, "Need an ode number of anchors points"
        axs = [
            [
                np.linspace(anchors[2 * k, d], anchors[2 * k + 1, d], n)
                for d in range(dim_to_interpolate)
            ]
            for k in range(int(nanchors / 2))
        ]
        linespce = []
        for i in range(len(axs)):
            crd = np.stack(
                [np.asarray(axs[i][j]) for j in range(dim_to_interpolate)], axis=0
            ).T
            coord = np.zeros((crd.shape[0], opts["zdim"]))
            coord[:, : crd.shape[1]] = crd
            linespce.append(coord)
        linespace = np.asarray(linespce)
    elif mode == "priors_interpolation":
        axs = [
            [
                np.linspace(anchors[0, d], anchors[k, d], n)
                for d in range(dim_to_interpolate)
            ]
            for k in range(1, nanchors)
        ]
        linespce = []
        for i in range(len(axs)):
            crd = np.stack(
                [np.asarray(axs[i][j]) for j in range(dim_to_interpolate)], axis=0
            ).T
            coord = np.zeros((crd.shape[0], opts["zdim"]))
            coord[:, : crd.shape[1]] = crd
            linespce.append(coord)
        linespace = np.asarray(linespce)
    else:
        assert False, "Unknown mode %s for vizualisation" % opts["mode"]
    return linespace
