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
from loss_functions import log_penalty, kl_penalty, vae_sigmoid_reconstruction_loss, moments_loss
from plot_functions import save_train, plot_sinkhorn, plot_embedded, save_latent_interpolation
from networks import encoder, decoder
from datahandler import datashapes

"""
# Path to inception model and stats for training set
sys.path.append('../TTUR')
sys.path.append('../inception')
import fid
inception_path = '../inception'
inception_model = os.path.join(inception_path, 'classify_image_graph_def.pb')
layername = 'FID_Inception_Net/pool_3:0'
"""

import pdb

class VAE(object):

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
            self.prior_params = np.concatenate([mean,Sigma],axis=0)
        elif opts['prior']=='dirichlet':
            self.prior_params = 0.5 * np.ones(opts['zdim'][-1],dtype='float32')
        else:
            assert False, 'Unknown prior %s' % opts['prior']

        # --- Initialize list container
        self.encoded, self.decoded = [], []
        self.samples = []
        self.match_penalties, self.match_penalty = [], 0.
        self.qz_params, self.pz_params = [], []

        # --- Encoding & decoding Loop
        encoded = self.points
        for n in range(opts['e_nlatents']):
            # - Encoding points
            # Setting output_dim
            enc_output_dim = 2*opts['zdim'][n]
            enc_mean, enc_Sigma = encoder(self.opts, input=encoded,
                                                archi=opts['e_arch'][n],
                                                num_layers=opts['e_nlayers'][n],
                                                num_units=opts['e_nfilters'][n],
                                                output_dim=enc_output_dim,
                                                scope='encoder/layer_%d' % (n+1),
                                                reuse=False,
                                                is_training=self.is_training)
            qz_params = tf.concat((enc_mean,enc_Sigma),axis=-1)
            self.qz_params.append(qz_params)
            encoded = sample_gaussian(opts, qz_params, 'tensorflow')
            self.encoded.append(encoded)
            if n<opts['e_nlatents']-1:
                self.match_penalty += log_penalty(encoded, enc_mean, enc_Sigma)
            else:
                self.match_penalty += kl_penalty(mean, Sigma, enc_mean, enc_Sigma)
            # - Decoding encoded points (i.e. reconstruct) & reconstruction cost
            if n==0:
                dec_mean, _ = decoder(self.opts, input=encoded,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                output_dim=2*np.prod(datashapes[opts['dataset']]),
                                                scope='decoder/layer_%d' % n,
                                                reuse=False,
                                                is_training=self.is_training)
                decoded = sample_bernoulli(dec_mean)
                decoded = tf.reshape(decoded,[-1]+datashapes[opts['dataset']])
                self.mean_decoded=tf.reshape(dec_mean,[-1]+datashapes[opts['dataset']])
                # Reconstruction loss
                self.loss_reconstruct = vae_sigmoid_reconstruction_loss(self.points,
                                                self.mean_decoded)
            else:
                # Setting output_dim
                dec_output_dim = 2*opts['zdim'][n-1]
                dec_mean, dec_Sigma = decoder(self.opts, input=encoded,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                output_dim=dec_output_dim,
                                                scope='decoder/layer_%d' % n,
                                                reuse=False,
                                                is_training=self.is_training)
                pz_params = tf.concat((dec_mean,dec_Sigma),axis=-1)
                self.pz_params.append(pz_params)
                decoded = sample_gaussian(opts, pz_params, 'tensorflow')
                # matching penalty (KL)
                self.match_penalty -= log_penalty(decoded,dec_mean,dec_Sigma)
            self.decoded.append(decoded)

        # --- Objectives, penalties, pretraining, FID
        # Compute objs
        self.objective = self.loss_reconstruct + self.match_penalty
        # Pre Training
        self.pretrain_loss()
        """
        # FID score
        self.blurriness = self.compute_blurriness()
        self.inception_graph = tf.Graph()
        self.inception_sess = tf.Session(graph=self.inception_graph)
        with self.inception_graph.as_default():
            self.create_inception_graph()
        self.inception_layer = self._get_inception_layer()
        """

        # --- Sampling from model (only for generation)
        samples = self.pz_samples
        for n in range(opts['nlatents']-1,-1,-1):
            if n==0:
                samples_mean, samples_Sigma = decoder(self.opts, input=samples,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                output_dim=2*np.prod(datashapes[opts['dataset']]),
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=self.is_training)
                samples = sample_bernoulli(samples_mean)
                samples = tf.reshape(samples,[-1]+datashapes[opts['dataset']])
            else:
                # Setting output_dim
                dec_output_dim = 2*opts['zdim'][n-1]
                # Reuse params
                if n>=opts['e_nlatents']:
                    reuse=False
                else:
                    reuse=True
                samples_mean, samples_Sigma = decoder(self.opts, input=samples,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                output_dim=dec_output_dim,
                                                scope='decoder/layer_%d' % n,
                                                reuse=reuse,
                                                is_training=self.is_training)
                p_params = tf.concat((samples_mean,samples_Sigma),axis=-1)
                samples = sample_gaussian(opts, p_params, 'tensorflow')
            self.samples.append(samples)

        # # --- full reconstructions latents layers
        # self.full_reconstruction()
        #
        # # --- full reconstructions from sampled encodings
        # self.multi_reconstruction()
        #
        # # --- Point interpolation for implicit-prior WAE (only for generation)
        # self.anchor_interpolation()

        # --- Optimizers, savers, etc
        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()

    def add_model_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        self.points = tf.placeholder(tf.float32, [None] + shape,
                                                name='points_ph')
        self.pz_samples = tf.placeholder(tf.float32, [None] + [opts['zdim'][-1],],
                                                name='noise_ph')
        self.anchors_points = tf.placeholder(tf.float32,
                                    [None] + [datashapes[opts['dataset']][-1]*opts['zdim'][0],],
                                    name='anchors_ph')

    def add_training_placeholders(self):
        opts = self.opts
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.lr_decay = decay
        self.is_training = is_training

    def full_reconstruction(self):
        # Reconstruct for each encoding layer
        opts = self.opts
        self.full_reconstructed = []
        for m in range(len(self.encoded)):
            reconstructed=self.encoded[m]
            for n in range(m,-1,-1):
                if n==0:
                    recon_mean, recon_Sigma = decoder(self.opts, input=reconstructed,
                                                    archi=opts['d_arch'][n],
                                                    num_layers=opts['d_nlayers'][n],
                                                    num_units=opts['d_nfilters'][n],
                                                    output_dim=2*np.prod(datashapes[opts['dataset']]),
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
                    # Setting output_dim
                    if opts['e_arch'][n-1]=='dcgan' or opts['e_arch'][n-1]=='dcgan_mod':
                        dec_output_dim = 2*datashapes[opts['dataset']][-1]*opts['zdim'][n-1]
                    else:
                        dec_output_dim = 2*opts['zdim'][n-1]
                    recon_mean, recon_Sigma = decoder(self.opts, input=reconstructed,
                                                    archi=opts['d_arch'][n],
                                                    num_layers=opts['d_nlayers'][n],
                                                    num_units=opts['d_nfilters'][n],
                                                    output_dim=dec_output_dim,
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
            # - Encoding points
            # Setting output_dim
            if opts['e_arch'][n]=='mlp' or n==opts['nlatents']-1:
                enc_output_dim = 2*opts['zdim'][n]
            else:
                enc_output_dim = 2*datashapes[opts['dataset']][-1]*opts['zdim'][n]
            enc_mean, enc_Sigma = encoder(self.opts, input=tf.expand_dims(encoded_samples[-1][0],0),
                                                archi=opts['e_arch'][n],
                                                num_layers=opts['e_nlayers'][n],
                                                num_units=opts['e_nfilters'][n],
                                                output_dim=enc_output_dim,
                                                scope='encoder/layer_%d' % (n+1),
                                                reuse=True,
                                                is_training=False)
            qz_params = tf.concat((enc_mean,enc_Sigma),axis=-1)
            qz_params = tf.concat([qz_params for i in range(20)],axis=0)
            enc_samples = sample_gaussian(opts, qz_params, 'tensorflow')
            encoded_samples.append(enc_samples)
        for m in range(len(encoded_samples[1:])):
            reconstructed=encoded_samples[m+1]
            for n in range(m,-1,-1):
                if n==0:
                    recon_mean, recon_Sigma = decoder(self.opts, input=reconstructed,
                                                    archi=opts['d_arch'][n],
                                                    num_layers=opts['d_nlayers'][n],
                                                    num_units=opts['d_nfilters'][n],
                                                    output_dim=2*np.prod(datashapes[opts['dataset']]),
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
                    # Setting output_dim
                    if opts['e_arch'][n-1]=='dcgan' or opts['e_arch'][n-1]=='dcgan_mod':
                        dec_output_dim = 2*datashapes[opts['dataset']][-1]*opts['zdim'][n-1]
                    else:
                        dec_output_dim = 2*opts['zdim'][n-1]
                    recon_mean, recon_Sigma = decoder(self.opts, input=reconstructed,
                                                    archi=opts['d_arch'][n],
                                                    num_layers=opts['d_nlayers'][n],
                                                    num_units=opts['d_nfilters'][n],
                                                    output_dim=dec_output_dim,
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
        anc_mean, anc_Sigma = decoder(self.opts, input=self.anchors_points,
                                        archi=opts['d_arch'][0],
                                        num_layers=opts['d_nlayers'][0],
                                        num_units=opts['d_nfilters'][0],
                                        output_dim=2*np.prod(datashapes[opts['dataset']]),
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
            batch_samples = sample_pz(opts, self.prior_params,
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
        logging.error('Training VAE %d latent layers\n' % opts['nlatents'])
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
        fixed_noise = sample_pz(opts, self.prior_params, npics)

        """
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
        """

        # Init all monitoring variables
        Loss, Loss_rec, Loss_rec_test = [], [], []
        Loss_match = []
        """
        mean_blurr, fid_scores = [], [],
        """
        decay, counter = 1., 0
        wait, wait_lambda = 0, 0
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
                batch_samples = sample_pz(opts, self.prior_params,
                                                opts['batch_size'])
                # Feeding dictionary
                feed_dict={self.points: batch_images,
                           self.pz_samples: batch_samples,
                           self.lr_decay: decay,
                           self.is_training: True}
                # Update encoder and decoder
                [_, loss, loss_rec, loss_match] = self.sess.run([
                                                self.wae_opt,
                                                self.objective,
                                                self.loss_reconstruct,
                                                self.match_penalty],
                                                feed_dict=feed_dict)
                Loss.append(-loss)
                Loss_rec.append(-loss_rec)
                Loss_match.append(-loss_match)

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
                                                       self.is_training:False})
                        loss_rec_test += l / batches_num_te
                    Loss_rec_test.append(-loss_rec_test)

                    # Auto-encoding test images & samples generated by the model
                    [reconstructed_test, encoded, samples] = self.sess.run(
                                                [self.decoded,
                                                 self.encoded,
                                                 self.samples],
                                                feed_dict={self.points:data.test_data[:30*npics],
                                                           self.pz_samples: fixed_noise,
                                                           self.is_training:False})

                    if opts['vizu_embedded'] and counter>1:
                        decoded = samples[:-1]
                        decoded = decoded[::-1]
                        decoded.append(fixed_noise)
                        plot_embedded(opts,encoded,decoded, #[fixed_noise,].append(samples)
                                                data.test_labels[:30*npics],
                                                work_dir,'embedded_e%04d_mb%05d.png' % (epoch, it))

                    # Auto-encoding training images
                    reconstructed_train = self.sess.run(self.decoded,
                                                feed_dict={self.points:data.data[200:200+npics],
                                                           self.is_training:False})

                    """
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
                    """

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

                    """
                    debug_str = 'FID=%.3f, BLUR=%10.4e' % (
                                                fid_scores[-1],
                                                mean_blurr[-1])
                    logging.error(debug_str)
                    """

                    print('')
                    # Making plots
                    if opts['save_train_data'] and epoch>=opts['epoch_num']-1:
                        save_train_data = True
                    else:
                        save_train_data = False

                    if opts['e_nlatents']==opts['nlatents']:
                        samples_prior = fixed_noise
                    else:
                        samples_prior = samples[opts['nlatents']-opts['e_nlatents']-1]

                    save_train(opts, data.data[200:200+npics], data.test_data[:npics],  # images
                                     data.test_labels[:30*npics],    # labels
                                     reconstructed_train[0], reconstructed_test[0][:npics], # reconstructions
                                     encoded[-1],   # encoded points (bottom)
                                     samples_prior, samples[-1],  # prior samples, model samples
                                     Loss, None, Loss_match, None,  # losses
                                     Loss_rec, Loss_rec_test,   # rec losses
                                     None,    # rec losses for each latents
                                     work_dir,  # working directory
                                     'res_e%04d_mb%05d.png' % (epoch, it),  # filename
                                     save_train_data) # save training data

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
                    if epoch >= 1000 and len(Loss_rec) > 0:
                        if wait_lambda > 1002 * batches_num:
                            opts['lambda'] = list(2*np.array(opts['lambda']))
                            wae_lambda = opts['lambda']
                            # last_rec = np.array(losses_rec)
                            # last_match = np.concatenate((last_rec[1:],abs(loss_match)*np.ones(1)),axis=0)
                            # new_lambda = 0.98 * np.array(wae_lambda) + \
                            #              0.02 * last_rec / last_match
                            # wae_lambda = list(new_lambda)
                            # opts['lambda'] = wae_lambda
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
        num_steps = 40
        num_anchors = 20
        imshape = datashapes[opts['dataset']]

        # --- Reconstructions
        logging.error('Encoding test images..')
        num_pics = 5000
        # [encoded,reconstructed] = self.sess.run([self.encoded,self.reconstructed[0]],
        encoded = self.sess.run(self.encoded,
                                feed_dict={self.points:data.test_data[:num_pics],
                                           self.is_training:False})
        # data_ids = np.random.choice(num_pics,20,replace=False)
        data_ids = np.arange(30,72)
        full_recon = self.sess.run(self.full_reconstructed,
                               feed_dict={self.points:data.test_data[data_ids],
                                          self.is_training: False})

        full_reconstructed = [data.test_data[data_ids],] + full_recon
        if opts['encoder'][0]=='gauss':
            data_ids = np.arange(53,54)
            sampled_recon = self.sess.run(self.sampled_reconstructed,
                                   feed_dict={self.points:data.test_data[data_ids],
                                              self.is_training: False})

            sampled_reconstructed = [np.concatenate([data.test_data[data_ids] for i in range(20)]),] + sampled_recon
        else:
            sampled_reconstructed = None

        # --- Encode anchors points and interpolate
        # inter_anchors = None
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
                                               self.is_training: False})
        else:
            dec_anchors = self.sess.run(self.decoded[-1],
                                    feed_dict={self.samples: np.reshape(enc_interpolation,[-1,]+encshape),
                                               self.is_training: False})
        inter_anchors = np.reshape(dec_anchors,[-1,num_int]+imshape)
        # adding data
        data_anchors = np.reshape(data_anchors,[-1,2]+imshape)
        inter_anchors = np.concatenate((np.expand_dims(data_anchors[:,0],axis=1),inter_anchors),axis=1)
        inter_anchors = np.concatenate((inter_anchors,np.expand_dims(data_anchors[:,1],axis=1)),axis=1)

        # --- Latent interpolation
        logging.error('Latent interpolation..')
        # if opts['prior']!='implicit':
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
                                           self.is_training: False})
        inter_latent = np.reshape(dec_latent,[-1,num_steps]+imshape)

        # --- Samples generation
        logging.error('Samples generation..')
        num_cols = 15
        npics = num_cols**2
        prior_noise = sample_pz(opts, self.prior_params, npics)
        samples = self.sess.run(self.decoded[-1],
                               feed_dict={self.samples: prior_noise,
                                          self.is_training: False})
        # --- Making & saving plots
        logging.error('Saving images..')
        save_latent_interpolation(opts, data.test_data[:num_pics],data.test_labels[:num_pics], # data,labels
                        encoded, #reconstructed, # encoded, reconstructed points
                        full_reconstructed, sampled_reconstructed, # full & sampled recons
                        inter_anchors, inter_latent, # anchors and latents interpolation
                        samples, # samples
                        MODEL_PATH) # working directory

    # def test(self, data, MODEL_DIR, WEIGHTS_FILE):
    #     """
    #     Test trained MoG model with chosen method
    #     """
    #     opts = self.opts
    #     # Load trained weights
    #     MODEL_PATH = os.path.join(opts['method'],MODEL_DIR)
    #     if not tf.gfile.IsDirectory(MODEL_PATH):
    #         raise Exception("model doesn't exist")
    #     WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
    #     if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
    #         raise Exception("weights file doesn't exist")
    #     self.saver.restore(self.sess, WEIGHTS_PATH)
    #     # Set up
    #     batch_size = 100
    #     tr_batches_num = int(data.num_points / batch_size)
    #     train_size = data.num_points
    #     te_batches_num = int(np.shape(data.test_data)[0] / batch_size)
    #     test_size = np.shape(data.test_data)[0]
    #     debug_str = 'test data size: %d' % (np.shape(data.test_data)[0])
    #     logging.error(debug_str)
    #
    #     ### Compute probs
    #     # Iterate over batches
    #     logging.error('Determining clusters ID using training..')
    #     mean_probs = np.zeros((10,10))
    #     for it in range(tr_batches_num):
    #         # Sample batches of data points and Pz noise
    #         data_ids = np.random.choice(train_size, opts['batch_size'],
    #                                             replace=True)
    #         batch_images = data.test_data[data_ids].astype(np.float32)
    #         batch_labels = data.test_labels[data_ids].astype(np.float32)
    #         pi_train = self.sess.run(self.pi, feed_dict={
    #                                             self.points:batch_images,
    #                                             self.is_training:False})
    #         mean_prob = get_mean_probs(opts,batch_labels,pi_train)
    #         mean_probs += mean_prob / tr_batches_num
    #     # Determine clusters given mean probs
    #     labelled_clusters = relabelling_mask_from_probs(opts, mean_probs)
    #     logging.error('Clusters ID:')
    #     print(labelled_clusters)
    #
    #     ### Accuracy
    #     logging.error('Computing losses & accuracy..')
    #     # Training accuracy & loss
    #     acc_tr = 0.
    #     loss_rec_tr, loss_match_tr = 0., 0.
    #     for it in range(tr_batches_num):
    #         # Sample batches of data points and Pz noise
    #         data_ids = np.random.choice(train_size, batch_size,
    #                                             replace=True)
    #         batch_images = data.data[data_ids].astype(np.float32)
    #         batch_labels = data.labels[data_ids].astype(np.float32)
    #         batch_mix_noise = sample_pz(opts, self.pz_mean,
    #                                             self.pz_cov,
    #                                             batch_size,
    #                                             sampling_mode='all_mixtures')
    #         # Accuracy & losses
    #         [loss_rec, loss_match, pi] = self.sess.run([self.loss_reconstruct,
    #                                             self.match_penalty,
    #                                             self.pi],
    #                                             feed_dict={self.points:batch_images,
    #                                                        self.sample_mix_noise: batch_mix_noise,
    #                                                        self.is_training:False})
    #         acc = accuracy(batch_labels,pi,labelled_clusters)
    #         acc_tr += acc / tr_batches_num
    #         loss_rec_tr += loss_rec / tr_batches_num
    #         loss_match_tr += loss_match / tr_batches_num
    #     # Testing accuracy and losses
    #     acc_te = 0.
    #     loss_rec_te, loss_match_te = 0., 0.
    #     for it in range(te_batches_num):
    #         # Sample batches of data points and Pz noise
    #         data_ids = np.random.choice(test_size,
    #                                     batch_size,
    #                                     replace=True)
    #         batch_images = data.test_data[data_ids].astype(np.float32)
    #         batch_labels = data.test_labels[data_ids].astype(np.float32)
    #         batch_mix_noise = sample_pz(opts, self.pz_mean,
    #                                             self.pz_cov,
    #                                             batch_size,
    #                                             sampling_mode='all_mixtures')
    #         # Accuracy & losses
    #         [loss_rec, loss_match, pi] = self.sess.run([self.loss_reconstruct,
    #                                             self.match_penalty,
    #                                             self.pi],
    #                                             feed_dict={self.points:batch_images,
    #                                                        self.sample_mix_noise: batch_mix_noise,
    #                                                        self.is_training:False})
    #         acc = accuracy(batch_labels,probs,labelled_clusters)
    #         acc_te += acc / tr_batches_num
    #         loss_rec_te += loss_rec / te_batches_num
    #         loss_match_te += loss_match / te_batches_num
    #
    #     ### Logs
    #     debug_str = 'rec train: %.4f, rec test: %.4f' % (loss_rec_tr,
    #                                                    loss_rec_te)
    #     logging.error(debug_str)
    #     debug_str = 'match train: %.4f, match test: %.4f' % (loss_match_tr,
    #                                                        loss_match_te)
    #     logging.error(debug_str)
    #     debug_str = 'acc train: %.2f, acc test: %.2f' % (100.*acc_tr,
    #                                                          100.*acc_te)
    #     logging.error(debug_str)
    #
    #     ### Saving
    #     filename = 'res_test'
    #     res_test = np.array((loss_rec_tr, loss_rec_te,
    #                         loss_match_tr, loss_match_te,
    #                         acc_tr, acc_te))
    #     np.save(os.path.join(MODEL_PATH,filename),res_test)
    #
    #
