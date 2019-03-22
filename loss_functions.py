import sys
import time
import os
from math import sqrt, cos, sin, pow
import numpy as np
import tensorflow as tf

import utils
from datahandler import datashapes
from ops._ops import logsumexp, logsumexp_v2

import pdb


def vae_matching_penalty(pz_mean, pz_sigma, encoded_mean, encoded_sigma):
    """
    Compute KL divergence between prior and variational distribution
    """
    kl = encoded_sigma / pz_sigma \
        + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
        + tf.log(pz_sigma) - tf.log(encoded_sigma)
    kl = 0.5 * tf.reduce_sum(kl,axis=-1)
    return tf.reduce_mean(kl)


def matching_penalty(opts, samples_pz, samples_qz):
    """
    Compute the WAE's matching penalty
    (add here other penalty if any)
    """
    if opts['penalty']=='mmd':
        macth_penalty = mmd_penalty(opts, samples_pz, samples_qz)
    elif opts['penalty']=='sinkhorn':
        macth_penalty = sinkhorn_penalty(opts, samples_pz, samples_qz)
    else:
        raise ValueError('Unknown matching penalty term')
    return macth_penalty


def sinkhorn_penalty(opts, samples_pz, samples_qz):
    """
    Compute the sinkhorn distance penatly as
    in Sinkhorn Auto Encoders
    """
    # Compute Cost matrix
    C = square_dist_v2(opts, samples_pz, samples_qz)
    # Sinkhorn fixed points iteration
    sinkhorn = sinkhorn_it_v2(opts, C)
    return sinkhorn[-1]


def sinkhorn_it(opts, C):
    # Batch size
    M = utils.get_batch_size(C)
    # Kernel
    log_K = - C / opts['epsilon']
    # Initialization
    log_v = - logsumexp(log_K, axis=1, keepdims=True)
    Sinkhorn = []
    # Sinkhorn iterations
    for l in range(opts['L']-1):
        log_u = - logsumexp(log_K + log_v, axis=0, keepdims=True)
        Sinkhorn.append(tf.reduce_sum(tf.exp(log_u+log_K+log_v) * C))
        log_v = - logsumexp(log_K + log_u, axis=1, keepdims=True)
    log_u = - logsumexp(log_K + log_v, axis=0, keepdims=True)
    Sinkhorn.append(tf.reduce_sum(tf.exp(log_u+log_K+log_v) * C))
    return Sinkhorn


def sinkhorn_it_v2(opts,C):
    # Batch size
    M = utils.get_batch_size(C)
    # Initialization
    u = opts['epsilon']*(tf.log(M) - logsumexp(-C / opts['epsilon'], axis=1, keepdims=True))
    v = opts['epsilon']*(tf.log(M) - logsumexp((-C + u)/opts['epsilon'], axis=0, keepdims=True))
    Sinkhorn = []
    sinkhorn_init = tf.reduce_sum(tf.exp((-C + u + v)/opts['epsilon']) * C)
    Sinkhorn.append(sinkhorn_init)
    # Sinkhorn iterations
    for l in range(opts['L']-1):
        u -= opts['epsilon']*(tf.log(M) + logsumexp((-C + u + v)/opts['epsilon'], axis=1, keepdims=True))
        v -= opts['epsilon']*(tf.log(M) + logsumexp((-C + u + v)/opts['epsilon'], axis=0, keepdims=True))
        Sinkhorn.append(tf.reduce_sum(tf.exp((-C + u + v)/opts['epsilon']) * C))
    return Sinkhorn


def mmd_penalty(opts, sample_qz, sample_pz):
    sigma2_p = opts['pz_scale'] ** 2
    kernel = opts['mmd_kernel']
    n = utils.get_batch_size(sample_qz)
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = (n * n - n) / 2

    distances_pz = square_dist(opts, sample_pz, sample_pz)
    distances_qz = square_dist(opts, sample_qz, sample_qz)
    distances = square_dist(opts, sample_qz, sample_pz)

    if opts['mmd_kernel'] == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        # Maximal heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
        # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
        # sigma2_k = opts['latent_space_dim'] * sigma2_p
        if opts['verbose']:
            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat = res1 - res2
    elif opts['mmd_kernel'] == 'IMQ':
        Cbase = 2 * opts['zdim'][-1] * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
    return stat


def square_dist(opts, sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1, keepdims=True)
    norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1, keepdims=True)

    squared_dist = norms_x + tf.transpose(norms_y) \
                    - 2. * tf.matmul(sample_x,sample_y,transpose_b=True)
    return tf.nn.relu(squared_dist)


def square_dist_v2(opts, sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    x = tf.expand_dims(sample_x,axis=1)
    y = tf.expand_dims(sample_y,axis=0)
    squared_dist = tf.reduce_sum(tf.square(x - y),axis=-1)
    return squared_dist


def kl_penalty(enc_mu,enc_Sigma, prior_mu, prior_Sigma):
    sigma_ratio = tf.divide(enc_Sigma,prior_Sigma)
    kl = 1. + tf.log(sigma_ratio) - sigma_ratio \
        - tf.divide(tf.square(prior_mu-enc_mu),prior_Sigma)
    kl = tf.reduce_sum(kl,axis=-1) / 2.
    kl = tf.reduce_mean(kl)
    return kl


def reconstruction_loss(opts, x1, x2):
    """
    Compute the WAE's reconstruction losses
    x1: image data             [batch,im_dim]
    x2: image reconstruction   [batch,im_dim]
    """
    # Flatten if necessary
    assert len(x1.get_shape().as_list())==len(x2.get_shape().as_list()), \
                'data and reconstruction must have the same shape'
    x1 = tf.layers.flatten(x1)
    x2 = tf.layers.flatten(x2)
    # Compute chosen cost
    if opts['cost'] == 'l2':
        # c(x,y) = ||x - y||_2
        cost = tf.reduce_sum(tf.square(x1 - x2), axis=-1)
        cost = tf.sqrt(1e-10 + cost)
    elif opts['cost'] == 'l2sq':
        # c(x,y) = sum_i(||x - y||_2^2[:,i])
        cost = tf.reduce_sum(tf.square(x1 - x2), axis=-1)
    elif opts['cost'] == 'l2sq_norm':
        # c(x,y) = mean_i(||x - y||_2^2[:,i])
        cost = tf.reduce_mean(tf.square(x1 - x2), axis=-1)
    elif opts['cost'] == 'l1':
        # c(x,y) = ||x - y||_1
        cost = tf.reduce_sum(tf.abs(x1 - x2), axis=-1)
    elif opts['cost']=='cross_entropy':
        assert False, 'To implement cost function %s' % opts['cost']
        # c(x,y) = sum_i x[i]log y[i] + (1-x[i](log 1-y[i])
        eps = 1e-5
        cost = tf.reduce_sum(x1*tf.log(eps+x2)+(1-x1)*tf.log(eps+1-x2),axis=-1)
    else:
        assert False, 'Unknown cost function %s' % opts['cost']
    # Compute loss
    loss = opts['coef_rec'] * tf.reduce_mean(cost) #coef: .2 for L2 and L1, .05 for L2sqr,
    return loss


def vae_reconstruction_loss(x1, x2):
    """
    Compute the VAE's reconstruction losses
    x1: image data             [batch,im_dim]
    x2: image reconstruction   [batch,im_dim]
    """
    eps = 1e-8
    l = x1*tf.log(eps+x2) + (1-x1)*tf.log(eps+1-x2)
    l = -tf.reduce_sum(l,axis=[1,2,3])
    return tf.reduce_mean(l)


def contrast_norm(pics):
    # pics is a [N, H, W, C] tensor
    mean, var = tf.nn.moments(pics, axes=[-3, -2, -1], keep_dims=True)
    return pics / tf.sqrt(var + 1e-08)


def moments_loss(prior_samples, model_samples):
    # Matching the first 2 moments (mean and covariance)
    # Means
    qz_means = tf.reduce_mean(model_samples, axis=0)
    pz_mean = tf.reduce_mean(prior_samples, axis=0)
    mean_loss = tf.reduce_mean(tf.square(qz_means - pz_mean))
    # Covariances
    qz_covs = tf.reduce_mean(tf.square(model_samples-qz_means),axis=0)
    pz_cov = tf.reduce_mean(tf.square(prior_samples-pz_mean),axis=0)
    cov_loss = tf.reduce_mean(tf.square(qz_covs - pz_cov))
    # Loss
    pre_loss = mean_loss + cov_loss
    return pre_loss
