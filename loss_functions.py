import sys
import time
import os
from math import sqrt, cos, sin, pow, pi
import numpy as np
import tensorflow as tf

import utils
from datahandler import datashapes
from ops._ops import logsumexp, logsumexp_v2

import pdb


### --- Latent penalty --- ###
def log_normal(z, mean, sigma, eps=1e-6):
    """
    Compute gaussian log-density
    """
    c = tf.compat.v1.log(2*pi)
    return -0.5*(c + tf.compat.v1.log(eps+sigma) + tf.square(z - mean) / (1e-5+sigma))

def KL(mu0, cov0, mu1, cov1, eps=1e-6):
    """
    Compute KL divergence between gaussian prior and variational distribution
    """
    kl = (cov0) / (eps+cov1) + tf.square(mu1 - mu0)/(eps+cov1) + tf.compat.v1.log((eps+cov1)/(eps+cov0)) - 1
    kl = 0.5 * tf.reduce_sum(kl,axis=-1)
    return tf.reduce_mean(kl)

def mc_kl_penalty(samples, q_mean, q_Sigma, p_mean, p_Sigma):
    """
    Compute MC log density ratio
    """
    kl = tf.compat.v1.log(q_Sigma) - tf.compat.v1.log(p_Sigma) \
        + tf.square(samples - q_mean) / q_Sigma \
        - tf.square(samples - p_mean) / p_Sigma
    kl = -0.5 * tf.reduce_sum(kl,axis=-1)
    return tf.reduce_mean(kl)

def latent_penalty(opts, samples_pz, samples_qz):
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
    u = opts['epsilon']*(tf.compat.v1.log(M) - logsumexp(-C / opts['epsilon'], axis=1, keepdims=True))
    v = opts['epsilon']*(tf.compat.v1.log(M) - logsumexp((-C + u)/opts['epsilon'], axis=0, keepdims=True))
    Sinkhorn = []
    sinkhorn_init = tf.reduce_sum(tf.exp((-C + u + v)/opts['epsilon']) * C)
    Sinkhorn.append(sinkhorn_init)
    # Sinkhorn iterations
    for l in range(opts['L']-1):
        u -= opts['epsilon']*(tf.compat.v1.log(M) + logsumexp((-C + u + v)/opts['epsilon'], axis=1, keepdims=True))
        v -= opts['epsilon']*(tf.compat.v1.log(M) + logsumexp((-C + u + v)/opts['epsilon'], axis=0, keepdims=True))
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
    elif opts['mmd_kernel'] == 'RQ':
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            res1 = (1. + distances_qz / scale / 2.) ** (-scale)
            res1 += (1. + distances_pz / scale / 2.) ** (-scale)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = (1. + distances / scale / 2.) ** (-scale)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2

    return stat

def square_dist(opts, sample_x, sample_y):
    if opts['sqrdist'] == 'dotprod':
        return square_dist_dotprod(opts, sample_x, sample_y)
    elif opts['sqrdist'] == 'broadcast':
        return square_dist_broadcast(opts, sample_x, sample_y)
    else:
        raise Exception('Unknown {} square distance'.format(self.opts['sqrdist']))

def square_dist_dotprod(opts, sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1, keepdims=True)
    norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1, keepdims=True)

    squared_dist = norms_x + tf.transpose(norms_y) \
                    - 2. * tf.matmul(sample_x,sample_y,transpose_b=True)
    return tf.nn.relu(squared_dist)

def square_dist_broadcast(opts, sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    x = tf.expand_dims(sample_x,axis=1)
    y = tf.expand_dims(sample_y,axis=0)
    squared_dist = tf.reduce_sum(tf.square(x - y),axis=-1)
    return squared_dist


### --- Reconstruction costs --- ###
def obs_reconstruction_loss(opts, x1, x2):
    """
    Compute the WAE's reconstruction losses for the top layer
    x1: image data             [batch,im_shape]
    x2: image reconstruction   [batch,nsamples,im_dim]
    """
    # assert len(x1.get_shape().as_list())==len(x2.get_shape().as_list()), \
    #             'data and reconstruction must have the same shape'
    # Flatten last dim input
    x1 = tf.compat.v1.layers.flatten(x1)
    # Expand dim x1 if needed and flatten last dim input
    if len(x2.get_shape().as_list())>2:
        x1 = tf.expand_dims(x1,axis=1)
    # Compute chosen cost
    if opts['obs_cost'] == 'l2':
        cost = l2_cost(x1, x2)
    elif opts['obs_cost'] == 'l2sq':
        cost = l2sq_cost(x1, x2)
    elif opts['obs_cost'] == 'l2sq_norm':
        cost = l2sq_norm_cost(x1, x2)
    elif opts['obs_cost'] == 'l1':
        cost = l1_cost(x1, x2)
    elif opts['obs_cost'] == 'cross_entropy':
        cost = cross_entropy_cost(x1, x2)
    else:
        assert False, 'Unknown cost function %s' % opts['obs_cost']
    # Compute loss
    loss = tf.reduce_mean(cost) #coef: .2 for L2 and L1, .05 for L2sqr in WAE
    return loss

def latent_reconstruction_loss(opts, x1, mu1, Sigma1, x2, mu2, Sigma2):
    """
    Compute the WAE's reconstruction losses for latent layers
    x1: image data              [batch,im_dim]
    x2: image reconstruction    [batch,nsamples,im_dim]
    mu: decoded mean            [batch,nsamples,im_dim]
    Sigma: decoded variance     [batch,nsamples,im_dim]
    """
    # assert len(x1.get_shape().as_list())==len(x2.get_shape().as_list()), \
    #             'data and reconstruction must have the same shape'
    # Expand dim x1 if needed
    if len(x1.get_shape().as_list())!=len(x2.get_shape().as_list()):
        x1 = tf.expand_dims(x1,axis=1)
    # Compute chosen cost
    if opts['latent_cost'] == 'l2':
        cost = l2_cost(x1, x2)
    elif opts['latent_cost'] == 'l2sq':
        cost = l2sq_cost(x1, x2)
    elif opts['latent_cost'] == 'l2sq_gauss':
        cost = l2sq_gauss_cost(x1, x2, mu2, Sigma2)
    elif opts['latent_cost'] == 'quad_norm_1':
        cost = quad_norm_1_cost(x1, x2, mu2, Sigma2)
    elif opts['latent_cost'] == 'l2sq_norm':
        cost = l2sq_norm_cost(x1, x2)
    elif opts['latent_cost'] == 'l1':
        cost = l1_cost(x1, x2)
    elif opts['latent_cost'] == 'mahalanobis':
        cost = mahalanobis_cost(x1, x2)
    elif opts['latent_cost'] == 'mahalanobis_v2':
        cost = mahalanobis_cost_v2(x1, x2, mu2, Sigma2)
    elif opts['latent_cost'] == 'mahalanobis_v3':
        cost = mahalanobis_cost_v3(x1, x2, mu2, Sigma2)
    else:
        assert False, 'Unknown cost function %s' % opts['obs_cost']
    # Compute loss
    loss = tf.reduce_mean(cost) #coef: .2 for L2 and L1, .05 for L2sqr in WAE
    return loss

def l2_cost(x1, x2):
    # c(x,y) = ||x - y||_2
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=-1)
    cost = tf.sqrt(1e-10 + cost)
    return cost

def l2sq_cost(x1,x2):
    # c(x,y) = sum_i(||x - y||_2^2[:,i])
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=-1)
    return cost

def l2sq_gauss_cost(x1, x2, mu, Sigma):
    # c(x,y) = sum_i(Sigma[i]+(mu[i]-x1)^2)
    cost = tf.reduce_sum(Sigma + tf.square(mu-x1),axis=-1)
    return cost

def quad_norm_1_cost(x1, x2, mu, Sigma):
    # c(x,y) = sum_i( (Sigma[i]+(mu[i]-x1)^2)/(1+Sigma[i]) )
    cost = Sigma + tf.square(mu-x1)
    cost = tf.divide(cost, 1. + Sigma)
    cost = tf.reduce_sum(cost,axis=-1)
    return cost

def mahalanobis_cost(x1, x2):
    Sigma1 = cov(x1)
    Sigma2 = cov(x2)
    Sigma = 0.5*(Sigma1+Sigma2)
    Sigma_inv = tf.matrix_inverse(Sigma)
    cost = tf.tensordot(Sigma_inv, tf.expand_dims(x1-x2,axis=-1), axes=[[1], [1]])
    cost = tf.transpose(cost, [1, 0, 2])
    cost = tf.matmul(tf.expand_dims(x1-x2,axis=1),cost)
    return tf.squeeze(cost,[1,2])

def mahalanobis_cost_v2(x1, x2, mu, Sigma):
    eps = 1e-10
    Sigma_square = tf.matrix_diag(Sigma+eps)
    Sigma_inv = tf.matrix_inverse(Sigma_square)
    xdiff = tf.expand_dims(x1-mu,axis=-1)
    cost = tf.matmul(Sigma_inv, xdiff)
    cost = tf.matmul(tf.transpose(xdiff,perm=[0,1,3,2]),cost) \
            + x1.get_shape().as_list()[-1]
    return tf.reduce_mean(tf.squeeze(cost,[2,3]),axis=-1)

def mahalanobis_cost_v3(x1, x2, mu, Sigma):
    eps = 1e-10
    Sigma_square = tf.matrix_diag(Sigma+eps)
    mu1 = tf.reduce_mean(x1, axis=0, keepdims=True)
    Sigma1 = tf.expand_dims(cov(x1,mu1,axis=0),axis=1)
    pool_Sigma = 0.5 * (Sigma_square + Sigma1)
    pool_Sigma_inv = tf.matrix_inverse(pool_Sigma)
    xdiff = tf.expand_dims(x1-mu,axis=-1)
    cost = tf.matmul(pool_Sigma_inv, xdiff)
    cost = tf.matmul(tf.transpose(xdiff,perm=[0,1,3,2]),cost)
    cost = tf.squeeze(cost,[2,3])
    cost += tf.linalg.trace(tf.matmul(pool_Sigma_inv,Sigma_square))
    return tf.reduce_mean(cost,axis=-1)

def cov(x,mu,axis):
    mx = tf.matmul(tf.transpose(mu,perm=[0,2,1]), mu)
    if axis==0:
        assert x.get_shape().as_list()[1]==1, \
                'Input shape[1] must be 1 when computing covariance on axis 0'
        vx = tf.matmul(tf.transpose(x,perm=[1,2,0]), tf.transpose(x,perm=[1,0,2])) \
                /tf.cast(tf.shape(x)[0], tf.float32)
        return vx - mx
    elif axis==1:
        vx = tf.matmul(tf.transpose(x,perm=[0,2,1]), x)/tf.cast(tf.shape(x)[1], tf.float32)
        return vx - mx
    else:
        assert False, 'Wrong axis value for covariance'

def l2sq_norm_cost(x1, x2):
    # c(x,y) = mean_i(||x - y||_2^2[:,i])
    cost = tf.reduce_mean(tf.square(x1 - x2), axis=-1)
    return cost
    # if len(x2.get_shape().as_list())>2:
    #     return tf.reduce_mean(cost,axis=1)
    # else:
    #     return cost

def l1_cost(x1, x2):
    # c(x,y) = ||x - y||_1
    cost = tf.reduce_sum(tf.abs(x1 - x2), axis=-1)
    return cost
    # if len(x2.get_shape().as_list())>2:
    #     return tf.reduce_mean(cost,axis=1)
    # else:
    #     return cost

def cross_entropy_cost(x1, x2):
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=x1, logits=x2)
    return tf.reduce_sum(cost, axis=-1)


### --- various losses --- ###
def vae_reconstruction_loss(x1, x2):
    """
    Compute the VAE's reconstruction losses
    x1: image data             [batch,im_dim]
    x2: image reconstruction   [batch,im_dim]
    """
    eps = 1e-8
    l = x1*tf.compat.v1.log(eps+x2) + (1-x1)*tf.compat.v1.log(eps+1-x2)
    l = -tf.reduce_sum(l,axis=[1,2,3])
    return tf.reduce_mean(l)

def vae_sigmoid_reconstruction_loss(x1, logits):
    """
    Compute the VAE's reconstruction losses
    x1: image data              [batch,im_dim]
    x2: mean reconstruction     [batch,im_dim]
    """
    l = tf.nn.sigmoid_cross_entropy_with_logits(labels=x1,logits=logits)
    l = tf.reduce_sum(l,axis=[1,2,3])
    return tf.reduce_mean(l)

def contrast_norm(pics):
    # pics is a [N, H, W, C] tensor
    mean, var = tf.nn.moments(pics, axes=[-3, -2, -1], keepdims=True)
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
