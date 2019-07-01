# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

"""
Wasserstein Auto-Encoder models
"""

import sys
import time
import os
import logging

from math import sqrt, cos, sin, pi
import numpy as np
import tensorflow as tf

import utils
from sampling_functions import sample_pz, sample_gaussian, sample_bernoulli, linespace
from loss_functions import matching_penalty, obs_reconstruction_loss, latent_reconstruction_loss, moments_loss
from loss_functions import sinkhorn_it, sinkhorn_it_v2, square_dist, square_dist_v2
from plot_functions import save_train, plot_sinkhorn, plot_embedded, plot_encSigma, save_latent_interpolation, save_vlae_experiment
from networks import encoder, decoder
from datahandler import datashapes

# Path to inception model and stats for training set
sys.path.append('../TTUR')
sys.path.append('../inception')
import fid
inception_path = '../inception'
inception_model = os.path.join(inception_path, 'classify_image_graph_def.pb')
layername = 'FID_Inception_Net/pool_3:0'

import pdb

class WAE(object):

    def __init__(self, opts):

        logging.error('Building the Tensorflow Graph')

        # --- model warning
        if opts['prior']=='dirichlet' and opts['encoder']=='gauss':
            logging.error('Warning, training a gaussian encoder with dirichlet prior')

        # --- Create session
        self.sess = tf.Session()
        self.opts = opts
        # Check len hyperparams
        assert len(opts['zdim'])==opts['nlatents'], \
                'Num zdim does match number of latents'

        # --- Data shape
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        # --- Placeholders
        self.add_model_placeholders()
        self.add_training_placeholders()

        # --- Initialize prior parameters
        if opts['prior']=='gaussian':
            mean = np.zeros(opts['zdim'][-1], dtype='float32')
            Sigma = np.ones(opts['zdim'][-1], dtype='float32')
            self.pz_params = np.concatenate([mean,Sigma],axis=0)
        elif opts['prior']=='dirichlet':
            self.pz_params = 0.5 * np.ones(opts['zdim'][-1],dtype='float32')
        else:
            assert False, 'Unknown prior %s' % opts['prior']

        # --- Initialize list container
        encSigmas_stats = []
        decSigmas_stats = []
        pen_enc_sigma, pen_dec_sigma = 0., 0.
        if opts['e_arch'][0]=='resnet_v3':
            self.features_dim = [self.data_shape,]
        else:
            self.features_dim = [[self.data_shape[0],self.data_shape[1],opts['e_nfilters'][0]],]
        self.encoded, self.reconstructed = [], []
        self.decoded = []
        self.losses_reconstruct = []
        self.loss_reconstruct = 0

        # --- Encoding & decoding Loop
        for n in range(opts['e_nlatents']):
            # - Encoding points
            if n==0:
                input = self.points
            else:
                input = self.encoded[-1]
            if n==opts['e_nlatents']-1:
                top_latent=True
            else:
                top_latent=False
            enc_mean, enc_Sigma, out_shape = encoder(self.opts, input=input,
                                                archi=opts['e_arch'][n],
                                                num_layers=opts['e_nlayers'][n],
                                                num_units=opts['e_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim=2*opts['zdim'][n],
                                                features_dim=self.features_dim[-1],
                                                resample=opts['e_resample'][n],
                                                top_latent=top_latent,
                                                scope='encoder/layer_%d' % (n+1),
                                                reuse=False,
                                                is_training=self.is_training,
                                                dropout_rate=self.dropout_rate)
            if opts['encoder'][n] == 'det':
                # - deterministic encoder
                encoded = enc_mean
                if n==opts['nlatents']-1 and opts['prior']=='dirichlet':
                    encoded = tf.nn.softmax(encoded)
                self.encoded.append(encoded)
            elif opts['encoder'][n] == 'gauss':
                # - gaussian encoder
                qz_params = tf.concat((enc_mean,enc_Sigma),axis=-1)
                if opts['rec_loss_resamples']=='encoder':
                    qz_params = tf.stack([qz_params for i in range(opts['rec_loss_nsamples'])],axis=1)
                    encoded = sample_gaussian(opts, qz_params, 'tensorflow')
                    self.encoded.append(encoded[:,0])
                    encoded = tf.squeeze(tf.concat(tf.split(encoded,opts['rec_loss_nsamples'],1),axis=0),[1])
                else:
                    encoded = sample_gaussian(opts, qz_params, 'tensorflow')
                    self.encoded.append(encoded)
                # Enc Sigma penalty
                if opts['pen_enc_sigma']:
                    pen_enc_sigma += opts['zdim'][n]*tf.reduce_mean(tf.reduce_sum(tf.abs(tf.log(enc_Sigma)),axis=-1))
                # Enc Sigma stats
                Sigma_tr = tf.reduce_mean(enc_Sigma,axis=-1)
                Smean, Svar = tf.nn.moments(Sigma_tr,axes=[0])
                Sstats = tf.stack([Smean,Svar],axis=-1)
                encSigmas_stats.append(Sstats)
            else:
                assert False, 'Unknown encoder %s' % opts['encoder']
            self.features_dim.append(out_shape)
            # - Decoding encoded points (i.e. reconstruct) & reconstruction cost
            if n==0:
                output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
            else:
                output_dim=[2*opts['zdim'][n-1],]
            if opts['d_arch'][n]=='resnet_v3':
                features_dim=self.features_dim[-1]
            else:
                features_dim=self.features_dim[-2]
            recon_mean, recon_Sigma = decoder(self.opts, input=encoded,
                                            archi=opts['d_arch'][n],
                                            num_layers=opts['d_nlayers'][n],
                                            num_units=opts['d_nfilters'][n],
                                            filter_size=opts['filter_size'][n],
                                            output_dim=output_dim,
                                            features_dim=features_dim,
                                            resample=opts['d_resample'][n],
                                            scope='decoder/layer_%d' % n,
                                            reuse=False,
                                            is_training=self.is_training,
                                            dropout_rate=self.dropout_rate)
            # - reshaping or resampling reconstruced
            if opts['decoder'][n] == 'det':
                # - deterministic decoder
                reconstructed = recon_mean
                if opts['encoder'][n] != 'det' and opts['rec_loss_resamples']=='encoder':
                    reconstructed_list = tf.split(reconstructed,opts['rec_loss_nsamples'])
                    reconstructed = tf.stack(reconstructed_list,axis=1)
            elif opts['decoder'][n] == 'gauss':
                # - gaussian decoder
                if opts['encoder'][n] != 'det' and opts['rec_loss_resamples']=='encoder':
                    p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
                    reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
                    reconstructed_list = tf.split(reconstructed,opts['rec_loss_nsamples'])
                    reconstructed = tf.stack(reconstructed_list,axis=1)
                    recon_mean_list = tf.split(recon_mean,opts['rec_loss_nsamples'])
                    recon_mean = tf.stack(recon_mean_list,axis=1)
                    recon_Sigma_list = tf.split(recon_Sigma,opts['rec_loss_nsamples'])
                    recon_Sigma = tf.stack(recon_Sigma_list,axis=1)
                elif opts['rec_loss_resamples']=='decoder':
                    p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
                    p_params = tf.stack([p_params for i in range(opts['rec_loss_nsamples'])],axis=1)
                    recon_mean, recon_Sigma = tf.split(p_params,num_or_size_splits=2,axis=-1)
                    reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
                    reconstructed_list = tf.split(reconstructed,opts['rec_loss_nsamples'],axis=1)
                    reconstructed = tf.squeeze(tf.stack(reconstructed_list,axis=2),[1])
                # Dec Sigma penalty
                if opts['pen_dec_sigma']:
                    pen_dec_sigma += opts['zdim'][n]*tf.reduce_mean(tf.reduce_sum(tf.abs(tf.log(recon_Sigma)),axis=-1))
                # Dec Sigma stats
                if len(recon_Sigma.get_shape().as_list())>2:
                    Sigma_tr = tf.reduce_mean(tf.reduce_mean(recon_Sigma,axis=-1),axis=-1)
                else:
                    Sigma_tr = tf.reduce_mean(recon_Sigma,axis=-1)
                Smean, Svar = tf.nn.moments(Sigma_tr,axes=[0])
                Sstats = tf.stack([Smean,Svar],axis=-1)
                decSigmas_stats.append(Sstats)
            else:
                assert False, 'Unknown encoder %s' % opts['decoder'][n]
            # - reconstruction loss
            if n==0:
                if opts['decoder'][n]!='bernoulli':
                    if opts['input_normalize_sym']:
                        reconstructed=tf.nn.tanh(reconstructed)
                    else:
                        reconstructed=tf.nn.sigmoid(reconstructed)
                if len(reconstructed.get_shape().as_list())>2:
                    reconstructed = tf.reshape(reconstructed,[-1,opts['rec_loss_nsamples']]+datashapes[opts['dataset']])
                    self.reconstructed.append(reconstructed[:,0])
                else:
                    reconstructed = tf.reshape(reconstructed,[-1]+datashapes[opts['dataset']])
                    self.reconstructed.append(reconstructed)
                loss_reconstruct = obs_reconstruction_loss(opts, self.points, reconstructed)
                self.loss_reconstruct += loss_reconstruct
            else:
                if len(reconstructed.get_shape().as_list())>2:
                    self.reconstructed.append(reconstructed[:,0])
                else:
                    self.reconstructed.append(reconstructed)
                loss_reconstruct = latent_reconstruction_loss(opts,
                                                self.encoded[-2],
                                                reconstructed,
                                                recon_mean,
                                                recon_Sigma)
                self.loss_reconstruct += self.lmbd[n-1] * loss_reconstruct

            self.losses_reconstruct.append(loss_reconstruct)
        self.encSigmas_stats = tf.stack(encSigmas_stats,axis=0)
        self.decSigmas_stats = tf.stack(decSigmas_stats,axis=0)

        # --- Sampling from model (only for generation)
        decoded = self.samples
        for n in range(opts['nlatents']-1,-1,-1):
            if opts['d_arch'][n]=='resnet_v3':
                features_dim=self.features_dim[n+1]
            else:
                features_dim=self.features_dim[n]
            if n==0:
                output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
                decoded_mean, decoded_Sigma = decoder(self.opts, input=decoded,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim =output_dim,
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][n],
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=self.is_training)
                if opts['decoder'][n] == 'det':
                    decoded = decoded_mean
                elif opts['decoder'][n] == 'gauss':
                    p_params = tf.concat((decoded_mean,decoded_Sigma),axis=-1)
                    decoded = sample_gaussian(opts, p_params, 'tensorflow')
                elif opts['decoder'][n] == 'bernoulli':
                    decoded = sample_bernoulli(decoded_mean)
                else:
                    assert False, 'Unknown encoder %s' % opts['decoder'][n]
                if opts['decoder'][n]!='bernoulli':
                    if opts['input_normalize_sym']:
                        decoded=tf.nn.tanh(decoded)
                    else:
                        decoded=tf.nn.sigmoid(decoded)
                decoded = tf.reshape(decoded,[-1]+datashapes[opts['dataset']])
            else:
                # Reuse params
                if n>=opts['e_nlatents']:
                    reuse=False
                else:
                    reuse=True
                decoded_mean, decoded_Sigma = decoder(self.opts, input=decoded,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim = [2*opts['zdim'][n-1],],
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][n],
                                                scope='decoder/layer_%d' % n,
                                                reuse=reuse,
                                                is_training=self.is_training)
                if opts['decoder'][n] == 'det':
                    decoded = decoded_mean
                elif opts['decoder'][n] == 'gauss':
                    p_params = tf.concat((decoded_mean,decoded_Sigma),axis=-1)
                    decoded = sample_gaussian(opts, p_params, 'tensorflow')
                else:
                    assert False, 'Unknown encoder %s' % opts['decoder'][n]
            self.decoded.append(decoded)

        # --- Objectives, penalties, pretraining, FID
        if opts['pen']=='wae':
            # Compute matching penalty cost for last layer
            if opts['e_nlatents']==opts['nlatents']:
                self.match_penalty = matching_penalty(opts, self.samples, self.encoded[-1])
                self.C = square_dist_v2(self.opts,self.samples, self.encoded[-1])
            else:
                self.match_penalty = matching_penalty(opts, self.decoded[opts['nlatents']-opts['e_nlatents']-1],
                                                self.encoded[-1])
                self.C = square_dist_v2(self.opts,self.decoded[opts['nlatents']-opts['e_nlatents']-1],
                                                self.encoded[-1])
            # Compute objs
            self.objective = self.loss_reconstruct \
                             + self.lmbd[-1] * self.match_penalty
        elif opts['pen']=='wae_mmd':
            # Compute matching penalty cost for all layer
            self.C = square_dist_v2(self.opts,self.samples, self.encoded[-1])
            self.match_penalty = 0
            for n in range(opts['nlatents']-1):
                self.match_penalty += self.lmbd[-1]* 0.01 * matching_penalty(opts, self.decoded[opts['nlatents']-(n+2)],
                                                self.encoded[n])
            self.match_penalty += self.lmbd[-1] * matching_penalty(opts, self.samples, self.encoded[-1])
            # Compute objs
            self.objective = self.loss_reconstruct + self.match_penalty
        else:
            assert False, 'Unknown penalty %s' % opts['pen']
        # Enc Sigma penalty
        if opts['pen_enc_sigma']:
            self.objective += opts['lambda_pen_enc_sigma']*pen_enc_sigma
        # Dec Sigma penalty
        if opts['pen_dec_sigma']:
            self.objective += opts['lambda_pen_dec_sigma']*pen_dec_sigma
        # Implicit losses
        if opts['nlatents']>1:
            self.imp_match_penalty = matching_penalty(opts, self.decoded[-2], self.encoded[0])
            self.imp_objective = self.losses_reconstruct[0] \
                                + self.lmbd[0] * self.imp_match_penalty
        else:
            self.imp_match_penalty = self.match_penalty
            self.imp_objective = self.objective

        # Logging info
        self.sinkhorn = sinkhorn_it_v2(self.opts, self.C)
        # Pre Training
        self.pretrain_loss()

        if opts['fid']:
            # FID score
            self.blurriness = self.compute_blurriness()
            self.inception_graph = tf.Graph()
            self.inception_sess = tf.Session(graph=self.inception_graph)
            with self.inception_graph.as_default():
                self.create_inception_graph()
            self.inception_layer = self._get_inception_layer()

        # --- full reconstructions latents layers
        self.full_reconstruction()

        # --- full reconstructions from sampled encodings
        self.multi_reconstruction()

        # --- Point interpolation for implicit-prior WAE (only for generation)
        self.anchor_interpolation()

        # --- Optimizers, savers, etc
        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()

    def add_model_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        self.points = tf.placeholder(tf.float32, [None] + shape,
                                                name='points_ph')
        self.samples = tf.placeholder(tf.float32, [None] + [opts['zdim'][-1],],
                                                name='noise_ph')
        if opts['d_arch'][0]=='resnet_v3' and opts['nlatents']!=1:
            if opts['e_resample'][0]=='down':
                self.anchors_points = tf.placeholder(tf.float32,
                                            [None] + [int(self.data_shape[0]/2)*int(self.data_shape[1]/2)*opts['zdim'][0],],
                                            name='anchors_ph')
            else:
                self.anchors_points = tf.placeholder(tf.float32,
                                            [None] + [self.data_shape[0]*self.data_shape[1]*opts['zdim'][0],],
                                            name='anchors_ph')
        else:
            self.anchors_points = tf.placeholder(tf.float32,
                                        [None] + [opts['zdim'][0],],
                                        name='anchors_ph')

    def add_training_placeholders(self):
        opts = self.opts
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        lmbda = tf.placeholder(tf.float32, name='lambda_ph')
        dropout = tf.placeholder(tf.float32, name='dropout_rate_ph')
        self.lr_decay = decay
        self.is_training = is_training
        self.lmbd = lmbda
        self.dropout_rate = dropout

    def full_reconstruction(self):
        # Reconstruct for each encoding layer
        opts = self.opts
        self.full_reconstructed = []
        for m in range(len(self.encoded)):
            reconstructed=self.encoded[m]
            for n in range(m,-1,-1):
                if opts['d_arch'][n]=='resnet_v3':
                    features_dim=self.features_dim[n+1]
                else:
                    features_dim=self.features_dim[n]
                if n==0:
                    output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
                    recon_mean, recon_Sigma = decoder(self.opts, input=reconstructed,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim = output_dim,
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][n],
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=False)
                    if opts['decoder'][n] == 'det':
                        reconstructed = recon_mean
                    elif opts['decoder'][n] == 'gauss':
                        p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
                        reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
                    elif opts['decoder'][n] == 'bernoulli':
                        reconstructed = sample_bernoulli(recon_mean)
                    else:
                        assert False, 'Unknown encoder %s' % opts['decoder'][n]
                    if opts['decoder'][n] != 'bernoulli':
                        if opts['input_normalize_sym']:
                            reconstructed=tf.nn.tanh(reconstructed)
                        else:
                            reconstructed=tf.nn.sigmoid(reconstructed)
                    reconstructed = tf.reshape(reconstructed,[-1]+datashapes[opts['dataset']])
                else:
                    recon_mean, recon_Sigma = decoder(self.opts, input=reconstructed,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim = [2*opts['zdim'][n-1],],
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][n],
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=False)
                    if opts['decoder'][n] == 'det':
                        reconstructed = recon_mean
                    elif opts['decoder'][n] == 'gauss':
                        p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
                        reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
                    else:
                        assert False, 'Unknown encoder %s' % opts['decoder'][n]
            self.full_reconstructed.append(reconstructed)

    def multi_reconstruction(self):
        # Reconstruc multi encodings sampled from q
        opts = self.opts
        self.sampled_reconstructed = []
        encoded_samples = [self.points,]
        for n in range(opts['e_nlatents']):
            if n==opts['e_nlatents']-1:
                top_latent=True
            else:
                top_latent=False
            enc_mean, enc_Sigma, out_shape = encoder(self.opts, input=tf.expand_dims(encoded_samples[-1][0],0),
                                                archi=opts['e_arch'][n],
                                                num_layers=opts['e_nlayers'][n],
                                                num_units=opts['e_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim=2*opts['zdim'][n],
                                                features_dim=self.features_dim[n],
                                                resample=opts['e_resample'][n],
                                                top_latent=top_latent,
                                                scope='encoder/layer_%d' % (n+1),
                                                reuse=True,
                                                is_training=False)
            qz_params = tf.concat((enc_mean,enc_Sigma),axis=-1)
            qz_params = tf.concat([qz_params for i in range(6)],axis=0)
            enc_samples = sample_gaussian(opts, qz_params, 'tensorflow')
            encoded_samples.append(enc_samples)
        for m in range(len(encoded_samples[1:])):
            reconstructed=encoded_samples[m+1]
            for n in range(m,-1,-1):
                if opts['d_arch'][n]=='resnet_v3':
                    features_dim=self.features_dim[n+1]
                else:
                    features_dim=self.features_dim[n]
                if n==0:
                    output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
                    recon_mean, recon_Sigma = decoder(self.opts, input=reconstructed,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim=output_dim,
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][n],
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=False)
                    if opts['decoder'][n] == 'det':
                        reconstructed = recon_mean
                    elif opts['decoder'][n] == 'gauss':
                        p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
                        reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
                    elif opts['decoder'][n] == 'bernoulli':
                        reconstructed = sample_bernoulli(recon_mean)
                    else:
                        assert False, 'Unknown encoder %s' % opts['decoder'][n]
                    if opts['decoder'][n] != 'bernoulli':
                        if opts['input_normalize_sym']:
                            reconstructed=tf.nn.tanh(reconstructed)
                        else:
                            reconstructed=tf.nn.sigmoid(reconstructed)
                    reconstructed = tf.reshape(reconstructed,[-1,]+datashapes[opts['dataset']])
                else:
                    recon_mean, recon_Sigma = decoder(self.opts, input=reconstructed,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim=[2*opts['zdim'][n-1],],
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][n],
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=False)
                    if opts['decoder'][n] == 'det':
                        reconstructed = recon_mean
                    elif opts['decoder'][n] == 'gauss':
                        p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
                        reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
                    else:
                        assert False, 'Unknown encoder %s' % opts['decoder'][n]
            self.sampled_reconstructed.append(reconstructed)

    def anchor_interpolation(self):
        # Anchor interpolation for 1-layer encoder
        opts = self.opts
        if opts['d_arch'][0]=='resnet_v3':
            features_dim=self.features_dim[1]
        else:
            features_dim=self.features_dim[0]
        output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
        anc_mean, anc_Sigma = decoder(self.opts, input=self.anchors_points,
                                                archi=opts['d_arch'][0],
                                                num_layers=opts['d_nlayers'][0],
                                                num_units=opts['d_nfilters'][0],
                                                filter_size=opts['filter_size'][0],
                                                output_dim=output_dim,
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][0],
                                                scope='decoder/layer_0',
                                                reuse=True,
                                                is_training=False)
        if opts['decoder'][0] == 'det':
            anchors_decoded = anc_mean
        elif opts['decoder'][0] == 'gauss':
            p_params = tf.concat((anc_mean,anc_Sigma),axis=-1)
            anchors_decoded = sample_gaussian(opts, p_params, 'tensorflow')
        elif opts['decoder'][0] == 'bernoulli':
            anchors_decoded = sample_bernoulli(anc_mean)
        else:
            assert False, 'Unknown encoder %s' % opts['decoder'][0]
        if opts['decoder'][0] != 'bernoulli':
            if opts['input_normalize_sym']:
                anchors_decoded=tf.nn.tanh(anchors_decoded)
            else:
                anchors_decoded=tf.nn.sigmoid(anchors_decoded)
        self.anchors_decoded = tf.reshape(anchors_decoded,[-1]+datashapes[opts['dataset']])

    def add_savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=2)
        self.saver = saver

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif opts['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(lr, beta1=opts['adam_beta1'])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        opts = self.opts
        # WAE optimizer
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='decoder')
        ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.wae_opt = opt.minimize(loss=self.objective, var_list=ae_vars)
        # Pretraining optimizer
        if opts['e_pretrain']:
            pre_opt = self.optimizer(0.001)
            self.pre_opt = pre_opt.minimize(loss=self.pre_loss, var_list=encoder_vars)

    def pretrain_loss(self):
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz(ZN) will try to match those of Pz(ZN)
        opts = self.opts
        if opts['e_nlatents']==opts['nlatents']:
            self.pre_loss = moments_loss(self.samples, self.encoded[-1])
        else:
            self.pre_loss = moments_loss(self.decoded[opts['nlatents']-opts['e_nlatents']-1], self.encoded[-1])

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = opts['e_pretrain_it']
        batch_size = opts['e_pretrain_sample_size']
        train_size = data.num_points
        for step in range(steps_max):
            data_ids = np.random.choice(train_size, batch_size,
                                               replace=False)
            batch_images = data.data[data_ids].astype(np.float32)
            batch_samples = sample_pz(opts, self.pz_params,
                                                batch_size)
            [_, pre_loss] = self.sess.run([self.pre_opt, self.pre_loss],
                                                feed_dict={
                                                self.points: batch_images,
                                                self.samples: batch_samples,
                                                self.is_training: True})
        logging.error('Pretraining the encoder done.')
        logging.error ('Loss after %d iterations: %.3f' % (steps_max,pre_loss))

    def compute_blurriness(self):
        images = self.points
        # First convert to greyscale
        if self.data_shape[-1] > 1:
            # We have RGB
            images = tf.image.rgb_to_grayscale(images)
        # Next convolve with the Laplace filter
        lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap_filter = lap_filter.reshape([3, 3, 1, 1])
        conv = tf.nn.conv2d(images, lap_filter, strides=[1, 1, 1, 1],
                                                padding='VALID')
        _, lapvar = tf.nn.moments(conv, axes=[1, 2, 3])
        return lapvar

    def create_inception_graph(self):
        # Create inception graph
        with tf.gfile.FastGFile( inception_model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString( f.read())
            _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')

    def _get_inception_layer(self):
        # Get inception activation layer (and reshape for batching)
        pool3 = self.inception_sess.graph.get_tensor_by_name(layername)
        ops_pool3 = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops_pool3):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != []:
                  shape = [s.value for s in shape]
                  new_shape = []
                  for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                      new_shape.append(None)
                    else:
                      new_shape.append(s)
                  o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
        return pool3


    def train(self, data, WEIGHTS_FILE):
        """
        Train top-down model with chosen method
        """

        opts = self.opts
        logging.error('Training WAE %d latent layers\n' % opts['nlatents'])
        #print('')
        work_dir = opts['work_dir']

        # Init sess and load trained weights if needed
        if opts['use_trained']:
            if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.init)
            if opts['e_pretrain']:
                logging.error('Pretraining the encoder\n')
                self.pretrain_encoder(data)
                print('')

        # Set up for training
        train_size = data.num_points
        batches_num = int(train_size/opts['batch_size'])
        npics = opts['plot_num_pics']
        fixed_noise = sample_pz(opts, self.pz_params, npics)

        if opts['fid']:
            # Load inception mean samples for train set
            trained_stats = os.path.join(inception_path, 'fid_stats.npz')
            # Load trained stats
            f = np.load(trained_stats)
            self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
            f.close()
            # Compute bluriness of real data
            real_blurr = self.sess.run(self.blurriness, feed_dict={
                                                self.points: data.data[:npics]})
            logging.error('Real pictures sharpness = %10.4e' % np.min(real_blurr))
            print('')

        # Init all monitoring variables
        Loss, imp_Loss = [], []
        Loss_rec, Losses_rec, Loss_rec_test = [], [], []
        Loss_match, imp_Match = [], []
        enc_Sigmas, dec_Sigmas = [], []
        mean_blurr, fid_scores = [], [],
        decay, counter = 1., 0
        wait, wait_lambda = 0, 0
        wae_lambda = opts['lambda']
        self.start_time = time.time()
        for epoch in range(opts['epoch_num']):
            # Saver
            if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
                self.saver.save(self.sess, os.path.join(
                                                work_dir,'checkpoints',
                                                'trained-wae'),
                                                global_step=counter)
            ##### TRAINING LOOP #####
            for it in range(batches_num):
                # Sample batches of data points and Pz noise
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                                replace=True)
                batch_images = data.data[data_ids].astype(np.float32)
                batch_samples = sample_pz(opts, self.pz_params,
                                                opts['batch_size'])
                # Feeding dictionary
                feed_dict={self.points: batch_images,
                           self.samples: batch_samples,
                           self.lr_decay: decay,
                           self.lmbd: wae_lambda,
                           self.dropout_rate: opts['dropout_rate'],
                           self.is_training: True}
                # Update encoder and decoder
                [_, loss, imp_loss, loss_rec, losses_rec, loss_match, imp_match] = self.sess.run([
                                                self.wae_opt,
                                                self.objective,
                                                self.imp_objective,
                                                self.loss_reconstruct,
                                                self.losses_reconstruct,
                                                self.match_penalty,
                                                self.imp_match_penalty],
                                                feed_dict=feed_dict)
                Loss.append(loss)
                imp_Loss.append(imp_loss)
                Loss_rec.append(loss_rec)
                losses_rec = list(np.array(losses_rec)*np.concatenate((np.ones(1),np.array(wae_lambda[:opts['e_nlatents']-1]))))
                Losses_rec.append(losses_rec)
                Loss_match.append(wae_lambda[-1]*loss_match)
                imp_Match.append(wae_lambda[0]*imp_match)
                if opts['vizu_encSigma']:
                    [enc_sigmastats,dec_sigmastats] = self.sess.run(
                                                [self.encSigmas_stats,
                                                self.decSigmas_stats],
                                                feed_dict=feed_dict)
                    enc_Sigmas.append(enc_sigmastats)
                    dec_Sigmas.append(dec_sigmastats)

                ##### TESTING LOOP #####
                if counter % opts['print_every']==0 or counter==100:
                    now = time.time()
                    batch_size_te = 200
                    test_size = np.shape(data.test_data)[0]
                    batches_num_te = int(test_size/batch_size_te)
                    # Test loss
                    loss_rec_test = 0.
                    for it_ in range(batches_num_te):
                        # Sample batches of data points
                        data_ids =  np.random.choice(test_size, batch_size_te,
                                                replace=True)
                        batch_images = data.test_data[data_ids].astype(np.float32)
                        l = self.sess.run(self.loss_reconstruct,
                                                feed_dict={self.points:batch_images,
                                                           self.lmbd: wae_lambda,
                                                           self.dropout_rate: 1.,
                                                           self.is_training:False})
                        loss_rec_test += l / batches_num_te
                    Loss_rec_test.append(loss_rec_test)

                    # Auto-encoding test images & samples generated by the model
                    [reconstructed_test, encoded, samples] = self.sess.run(
                                                [self.full_reconstructed[-1],
                                                 self.encoded,
                                                 self.decoded],
                                                feed_dict={self.points:data.test_data[:10*npics],
                                                           self.samples: fixed_noise,
                                                           self.dropout_rate: 1.,
                                                           self.is_training:False})

                    if opts['vizu_embedded'] and counter>1:
                        decoded = samples[:-1]
                        decoded = decoded[::-1]
                        # decoded = samples[::-1]
                        decoded.append(fixed_noise)
                        plot_embedded(opts,encoded,decoded, #[fixed_noise,].append(samples)
                                                data.test_labels[:10*npics],
                                                work_dir,'embedded_e%04d_mb%05d.png' % (epoch, it))
                    if opts['vizu_sinkhorn']:
                        [C,sinkhorn] = self.sess.run([self.C, self.sinkhorn],
                                                feed_dict={self.points:data.test_data[:npics],
                                                           self.samples: fixed_noise,
                                                           self.dropout_rate: 1.,
                                                           self.is_training:False})
                        plot_sinkhorn(opts, sinkhorn, work_dir,
                                                'sinkhorn_e%04d_mb%05d.png' % (epoch, it))
                    if opts['vizu_encSigma'] and counter>1:
                        plot_encSigma(opts, enc_Sigmas, dec_Sigmas,
                                                work_dir,
                                                'encSigma_e%04d_mb%05d.png' % (epoch, it))


                    # Auto-encoding training images
                    reconstructed_train = self.sess.run(self.full_reconstructed[-1],
                                                feed_dict={self.points:data.data[200:200+npics],
                                                           self.dropout_rate: 1.,
                                                           self.is_training:False})

                    if opts['fid']:
                        # Compute FID score
                        gen_blurr = self.sess.run(self.blurriness,
                                                feed_dict={self.points: samples})
                        mean_blurr.append(np.min(gen_blurr))
                        # First convert to RGB
                        if np.shape(flat_samples)[-1] == 1:
                            # We have greyscale
                            flat_samples = self.sess.run(tf.image.grayscale_to_rgb(flat_samples))
                        preds_incep = self.inception_sess.run(self.inception_layer,
                                      feed_dict={'FID_Inception_Net/ExpandDims:0': flat_samples})
                        preds_incep = preds_incep.reshape((npics,-1))
                        mu_gen = np.mean(preds_incep, axis=0)
                        sigma_gen = np.cov(preds_incep, rowvar=False)
                        fid_score = fid.calculate_frechet_distance(mu_gen,
                                                sigma_gen,
                                                self.mu_train,
                                                self.sigma_train,
                                                eps=1e-6)
                        fid_scores.append(fid_score)

                    # Printing various loss values
                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (
                                                epoch + 1, opts['epoch_num'],
                                                it + 1, batches_num)
                    logging.error(debug_str)
                    debug_str = 'TRAIN LOSS=%.3f' % (Loss[-1])
                    logging.error(debug_str)
                    debug_str = 'REC=%.3f, REC TEST=%.3f, MATCH=%10.3e\n ' % (
                                                Loss_rec[-1],
                                                Loss_rec_test[-1],
                                                Loss_match[-1])
                    logging.error(debug_str)
                    if opts['vizu_sinkhorn']:
                        debug_str = 'mdist=%10.3e, Mdist=%10.3e, avgdist=%10.3e\n ' % (
                                                np.amin(C),
                                                np.amax(C),
                                                np.mean(C))
                        logging.error(debug_str)
                    if opts['fid']:
                        debug_str = 'FID=%.3f, BLUR=%10.4e' % (
                                                fid_scores[-1],
                                                mean_blurr[-1])
                        logging.error(debug_str)

                    print('')
                    # Making plots
                    if opts['e_nlatents']==opts['nlatents']:
                        samples_prior = fixed_noise
                    else:
                        samples_prior = samples[opts['nlatents']-opts['e_nlatents']-1]

                    save_train(opts, data.data[200:200+npics], data.test_data[:npics],  # images
                                     data.test_labels[:10*npics],    # labels
                                     reconstructed_train, reconstructed_test[:npics], # reconstructions
                                     encoded[-1],   # encoded points (bottom)
                                     samples_prior, samples[-1],  # prior samples, model samples
                                     Loss, Loss_match,  # losses
                                     Loss_rec, Losses_rec,   # rec losses
                                     work_dir,  # working directory
                                     'res_e%04d_mb%05d.png' % (epoch, it))  # filename

                # Update learning rate if necessary and counter
                # First 20 epochs do nothing
                if epoch >= 10000:
                    # If no significant progress was made in last 20 epochs
                    # then decrease the learning rate.
                    if np.mean(Loss_rec[-20:]) < np.mean(Loss_rec[-20 * batches_num:])-1.*np.var(Loss_rec[-20 * batches_num:]):
                        wait = 0
                    else:
                        wait += 1
                    if wait > 20 * batches_num:
                        decay = max(decay  / 1.33, 1e-6)
                        logging.error('Reduction in lr: %f\n' % decay)
                        print('')
                        wait = 0

                # Update regularizer if necessary
                if opts['lambda_schedule'] == 'adaptive':
                    if epoch >= .0 and len(Loss_rec) > 0:
                        if wait_lambda > 200 * batches_num + 1:
                            # opts['lambda'] = list(2*np.array(opts['lambda']))
                            opts['lambda'][-1] = 2*opts['lambda'][-1]
                            wae_lambda = opts['lambda']
                            logging.error('Lambda updated to %s\n' % wae_lambda)
                            print('')
                            wait_lambda = 0
                        else:
                            wait_lambda+=1

                counter += 1

        # Save the final model
        if opts['save_final'] and epoch > 0:
            self.saver.save(self.sess, os.path.join(work_dir,
                                                'checkpoints',
                                                'trained-wae-final'),
                                                global_step=counter)
        # save training data
        if opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(work_dir,data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path,name),
                        data_test=data.data[200:200+npics], data_train=data.test_data[:npics],
                        label_test=data.test_labels[:10*npics],
                        encoded = encoded[-1],
                        enc_Sigmas=np.array(enc_Sigmas),dec_Sigmas=np.array(dec_Sigmas),
                        rec_train=reconstructed_train, rec_test=reconstructed_test[:npics],
                        samples_prior=samples_prior, samples=samples[-1],
                        loss=np.array(Loss), imp_loss=np.array(imp_Loss),
                        loss_match=np.array(Loss_match), imp_match=np.array(imp_Match),
                        loss_rec=np.array(Loss_rec),
                        loss_rec_test=np.array(Loss_rec_test),
                        losses_rec=np.array(Losses_rec))

    def latent_interpolation(self, data, MODEL_PATH, WEIGHTS_FILE):
        """
        Plot and save different latent interpolation
        """

        opts = self.opts

        # --- Load trained weights
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)
        # Set up
        test_size = np.shape(data.test_data)[0]
        num_steps = 14
        num_anchors = 20
        imshape = datashapes[opts['dataset']]

        # --- Reconstructions
        logging.error('Encoding test images..')
        num_pics = 2000
        encoded, full_recons = self.sess.run([self.encoded,
                                # self.full_reconstructed[-1]],
                                self.full_reconstructed],
                                feed_dict={self.points:data.test_data[:num_pics],
                                           self.dropout_rate: 1.,
                                           self.is_training:False})
        reconstructed = full_recons[-1]
        data_ids = np.arange(30,30+42)
        full_recon = full_recons[data_ids]
        data_ids = np.arange(30,30+42)
        # full_recon = self.sess.run(self.full_reconstructed,
        #                        feed_dict={self.points:data.test_data[data_ids],
        #                                   self.dropout_rate: 1.,
        #                                   self.is_training: False})

        full_reconstructed = [data.test_data[data_ids],] + full_recon
        if opts['encoder'][0]=='gauss':
            data_ids = np.arange(36,37)
            sampled_recon = self.sess.run(self.sampled_reconstructed,
                                   feed_dict={self.points:data.test_data[data_ids],
                                              self.dropout_rate: 1.,
                                              self.is_training: False})

            sampled_reconstructed = [np.concatenate([data.test_data[data_ids] for i in range(6)]),] + sampled_recon
        else:
            sampled_reconstructed = None

        # --- Encode anchors points and interpolate
        logging.error('Anchors interpolation..')
        encshape = list(np.shape(encoded[-1])[1:])
        #anchors_ids = np.random.choice(num_pics,2*num_anchors,replace=False)
        anchors_ids = np.arange(2*num_anchors,3*num_anchors)
        data_anchors = data.test_data[anchors_ids]
        enc_anchors = np.reshape(encoded[-1][anchors_ids],[-1,2]+encshape)
        enc_interpolation = linespace(opts, num_steps, anchors=enc_anchors)
        num_int = np.shape(enc_interpolation)[1]
        if opts['e_nlatents']!=opts['nlatents']:
            dec_anchors = self.sess.run(self.anchors_decoded,
                                    feed_dict={self.anchors_points: np.reshape(enc_interpolation,[-1,]+encshape),
                                               self.dropout_rate: 1.,
                                               self.is_training: False})
        else:
            dec_anchors = self.sess.run(self.decoded[-1],
                                    feed_dict={self.samples: np.reshape(enc_interpolation,[-1,]+encshape),
                                               self.dropout_rate: 1.,
                                               self.is_training: False})
        inter_anchors = np.reshape(dec_anchors,[-1,num_int]+imshape)
        # adding data
        data_anchors = np.reshape(data_anchors,[-1,2]+imshape)
        inter_anchors = np.concatenate((np.expand_dims(data_anchors[:,0],axis=1),inter_anchors),axis=1)
        inter_anchors = np.concatenate((inter_anchors,np.expand_dims(data_anchors[:,1],axis=1)),axis=1)


        if opts['zdim'][-1]==2:
            # --- Latent interpolation
            logging.error('Latent interpolation..')
            if False:
                enc_mean = np.mean(encoded[-1],axis=0)
                enc_var = np.mean(np.square(encoded[-1]-enc_mean),axis=0)
            else:
                enc_mean = np.zeros(opts['zdim'][-1], dtype='float32')
                enc_var = np.ones(opts['zdim'][-1], dtype='float32')
            mins, maxs = enc_mean - 2.*np.sqrt(enc_var), enc_mean + 2.*np.sqrt(enc_var)
            x = np.linspace(mins[0], maxs[0], num=num_steps, endpoint=True)
            xymin = np.stack([x,mins[1]*np.ones(num_steps)],axis=-1)
            xymax = np.stack([x,maxs[1]*np.ones(num_steps)],axis=-1)
            latent_anchors = np.stack([xymin,xymax],axis=1)
            grid_interpolation = linespace(opts, num_steps,
                                    anchors=latent_anchors)
            dec_latent = self.sess.run(self.decoded[-1],
                                    feed_dict={self.samples: np.reshape(grid_interpolation,[-1,]+list(np.shape(enc_mean))),
                                               self.dropout_rate: 1.,
                                               self.is_training: False})
            inter_latent = np.reshape(dec_latent,[-1,num_steps]+imshape)
        else:
            inter_latent = None

        # --- Samples generation
        logging.error('Samples generation..')
        num_cols = 14
        npics = num_cols**2
        prior_noise = sample_pz(opts, self.pz_params, npics)
        samples = self.sess.run(self.decoded[-1],
                               feed_dict={self.samples: prior_noise,
                                          self.dropout_rate: 1.,
                                          self.is_training: False})
        # --- Making & saving plots
        logging.error('Saving images..')
        save_latent_interpolation(opts, data.test_data[:num_pics],data.test_labels[:num_pics], # data,labels
                        encoded, reconstructed[:npics], # encoded, reconstructed points
                        full_reconstructed, sampled_reconstructed, # full & sampled recons
                        inter_anchors, inter_latent, # anchors and latents interpolation
                        samples, # samples
                        MODEL_PATH) # working directory

    def vlae_experiment(self, data, MODEL_PATH, WEIGHTS_FILE):
        """
        Plot and save different latent interpolation
        """

        opts = self.opts
        # num_pics = opts['plot_num_pics']
        num_pics = 16

        # --- Sampling fixed noise
        fixed_noise = []
        for n in range(opts['nlatents']):
            mean = np.zeros(opts['zdim'][opts['nlatents']-1-n], dtype='float32')
            Sigma = np.ones(opts['zdim'][opts['nlatents']-1-n], dtype='float32')
            params = np.concatenate([mean,Sigma],axis=0)
            fixed_noise.append(sample_gaussian(opts, params, batch_size=1))

        # --- Decoding loop
        self.vlae_decoded = []
        for m in range(opts['nlatents']):
            if m==0:
                decoded = tf.convert_to_tensor(sample_pz(opts,self.pz_params,num_pics-1),
                                                dtype=tf.float32)
                decoded = tf.concat([tf.convert_to_tensor(fixed_noise[0],dtype=tf.float32),decoded],axis=0)
            else:
                decoded = tf.concat([fixed_noise[0] for i in range(num_pics)],axis=0)
            for n in range(opts['nlatents']-1,-1,-1):
                # Output dim
                if n==0:
                    output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
                else:
                    output_dim = [2*opts['zdim'][n-1],]
                if opts['d_arch'][n]=='resnet_v3':
                    features_dim=self.features_dim[n+1]
                else:
                    features_dim=self.features_dim[n]
                # Decoding
                decoded_mean, decoded_Sigma = decoder(opts, input=decoded,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim=output_dim,
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][n],
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=False)
                if opts['decoder'][n] == 'det':
                    decoded = decoded_mean
                elif opts['decoder'][n] == 'gauss':
                    if n==opts['nlatents']-m:
                        p_params = tf.concat((decoded_mean,decoded_Sigma),axis=-1)
                        decoded = sample_gaussian(opts, p_params, 'tensorflow')
                    else:
                        decoded =  decoded_mean + tf.multiply(fixed_noise[opts['nlatents']-n],tf.sqrt(1e-10+decoded_Sigma))
                else:
                    assert False, 'Unknown encoder %s' % opts['decoder'][n]
                # reshape and normalize for last decoding
                if n==0:
                    if opts['input_normalize_sym']:
                        decoded=tf.nn.tanh(decoded)
                    else:
                        decoded=tf.nn.sigmoid(decoded)
                    decoded = tf.reshape(decoded,[-1]+datashapes[opts['dataset']])
            self.vlae_decoded.append(decoded)

        # --- Load trained weights
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # --- vlae decoding
        decoded = self.sess.run(self.vlae_decoded,feed_dict={})

        # --- Making & saving plots
        logging.error('Saving images..')
        save_vlae_experiment(opts, decoded, MODEL_PATH)

    def fid_score(self, data, MODEL_PATH, WEIGHTS_FILE):
        """
        Compute FID score
        """

        opts = self.opts

        # --- Load trained weights
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # Setup
        batch_size = 1000
        batches_num = 1

        # Load inception mean samples for train set
        trained_stats = os.path.join(inception_path, 'fid_stats.npz')
        # Load trained stats
        f = np.load(trained_stats)
        self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
        f.close()
        # Compute bluriness of real data
        real_blurr = self.sess.run(self.blurriness, feed_dict={
                                                self.points: data.test_data[:batch_size]})
        real_blurr = np.mean(real_blurr)
        # logging.error('Real pictures sharpness = %10.4e' % np.min(real_blurr))
        # print('')

        # Test loop
        now = time.time()
        mean_blurr, fid_scores = 0., 0.
        for it_ in range(batches_num):
            # Samples
            noise = sample_pz(opts, self.pz_params, batch_size)
            # Sampling
            samples = self.sess.run(self.decoded[-1],
                                        feed_dict={self.samples: noise,
                                        self.is_training:False})
            # compute blur
            gen_blurr = self.sess.run(self.blurriness,
                                        feed_dict={self.points: samples})
            mean_blurr+= (np.mean(gen_blurr) / batches_num)
            # Compute FID score
            # First convert to RGB
            if np.shape(samples)[-1] == 1:
                # We have greyscale
                samples = self.sess.run(tf.image.grayscale_to_rgb(samples))
            preds_incep = self.inception_sess.run(self.inception_layer,
                          feed_dict={'FID_Inception_Net/ExpandDims:0': samples})
            preds_incep = preds_incep.reshape((batch_size,-1))
            mu_gen = np.mean(preds_incep, axis=0)
            sigma_gen = np.cov(preds_incep, rowvar=False)
            fid_score = fid.calculate_frechet_distance(mu_gen,
                                        sigma_gen,
                                        self.mu_train,
                                        self.sigma_train,
                                        eps=1e-6)
            fid_scores+= (fid_score / batches_num)

            # Logging
            debug_str = 'FID=%.3f, BLUR=%10.3e, REAL BLUR=%10.3e' % (
                                    fid_scores,
                                    mean_blurr,
                                    real_blurr)
            logging.error(debug_str)
        name = 'fid'
        np.savez(os.path.join(work_dir,name),
                    fid=np.array(fid_scores),
                    blur=np.array(mean_blurr))

    def test_losses(self, data, MODEL_PATH, WEIGHTS_FILE):
        """
        Compute losses
        """

        opts = self.opts

        # --- Load trained weights
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)
        work_dir = opts['work_dir']

        # Setup
        now = time.time()
        batch_size_te = 200
        test_size = np.shape(data.test_data)[0]
        batches_num_te = int(test_size/batch_size_te)
        train_size = data.num_points
        # Logging stat
        debug_str = 'TEST SIZE=%d, BATCH NUM=%d' % (
                                test_size,
                                batches_num_te)
        logging.error(debug_str)

        # Test loss
        wae_lambda = opts['lambda']
        loss, loss_test = 0., 0.
        loss_rec, loss_rec_test = 0., 0.
        loss_match, loss_match_test = 0., 0.
        for it_ in range(batches_num_te):
            # Sample batches of test data points
            data_ids =  np.random.choice(test_size, batch_size_te,
                                    replace=True)
            batch_images = data.test_data[data_ids].astype(np.float32)
            batch_samples = sample_pz(opts, self.pz_params,
                                            batch_size_te)
            l, lrec, lmatch = self.sess.run([self.objective,
                                                self.loss_reconstruct,
                                                self.match_penalty],
                                feed_dict={self.points:batch_images,
                                                self.samples: batch_samples,
                                                self.lmbd: wae_lambda,
                                                self.is_training:False})
            loss_test += l / batches_num_te
            loss_rec_test += lrec / batches_num_te
            loss_match_test += lmatch / batches_num_te
            # Sample batches of train data points
            data_ids =  np.random.choice(train_size, batch_size_te,
                                    replace=True)
            batch_images = data.data[data_ids].astype(np.float32)
            batch_samples = sample_pz(opts, self.pz_params,
                                            batch_size_te)
            l, lrec, lmatch = self.sess.run([self.objective,
                                                self.loss_reconstruct,
                                                self.match_penalty],
                                feed_dict={self.points:batch_images,
                                                self.samples: batch_samples,
                                                self.lmbd: wae_lambda,
                                                self.is_training:False})
            loss += l / batches_num_te
            loss_rec += lrec / batches_num_te
            loss_match += lmatch / batches_num_te

        # Logging
        debug_str = 'TRAIN: LOSS=%.3f, REC=%.3f, MATCH=%10.3e' % (
                                loss,
                                loss_rec,
                                loss_match)
        logging.error(debug_str)
        debug_str = 'TEST: LOSS=%.3f, REC=%.3f, MATCH=%10.3e' % (
                                loss_test,
                                loss_rec_test,
                                loss_match_test)
        logging.error(debug_str)

        name = 'losses'
        np.savez(os.path.join(work_dir,name),
                    loss_train=np.array(loss),
                    loss_rec_train=np.array(loss_rec),
                    loss_match_train=np.array(loss_match),
                    loss_test=np.array(loss_test),
                    loss_rec_test=np.array(loss_rec_test),
                    loss_match_test=np.array(loss_match_test))
