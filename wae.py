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
from sampling_functions import sample_pz, sample_gaussian, linespace
from loss_functions import matching_penalty, reconstruction_loss, moments_loss
from loss_functions import sinkhorn_it, sinkhorn_it_v2, square_dist, square_dist_v2
from plot_functions import save_train, plot_sinkhorn, plot_embedded, plot_encSigma, save_latent_interpolation
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
        assert len(opts['lambda'])==opts['nlatents'], \
                'Num lambdas does match number of latents'
        assert len(opts['zdim'])==opts['nlatents'], \
                'Num zdim does match number of latents'

        # --- Some of the parameters for future use
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        # --- Placeholders
        self.add_model_placeholders()
        self.add_training_placeholders()
        sample_size = tf.shape(self.points,out_type=tf.int32)[0]

        # --- Initialize prior parameters
        if opts['prior']=='gaussian' or opts['prior']=='implicit':
            mean = np.zeros(opts['zdim'][-1], dtype='float32')
            Sigma = np.ones(opts['zdim'][-1], dtype='float32')
            self.pz_params = np.concatenate([mean,Sigma],axis=0)
        elif opts['prior']=='dirichlet':
            self.pz_params = 0.5 * np.ones(opts['zdim'][-1],dtype='float32')
        else:
            assert False, 'Unknown prior %s' % opts['prior']

        # --- Initialize list container
        encSigmas_stats = []
        self.encoded, self.reconstructed = [], []
        self.decoded = []
        self.losses_reconstruct = []
        self.loss_reconstruct = 0

        # --- Encoding & decoding Loop
        encoded = self.points
        for n in range(opts['nlatents']):
            if (opts['prior']=='implicit' and n==0) or opts['prior']!='implicit':
                # - Encoding points
                # Setting output_dim
                if opts['e_arch'][n]=='mlp' or n==opts['nlatents']-1:
                    enc_output_dim = 2*opts['zdim'][n]
                else:
                    enc_output_dim = 2*datashapes[opts['dataset']][-1]*opts['zdim'][n]
                enc_mean, enc_Sigma = encoder(self.opts, input=encoded,
                                                    archi=opts['e_arch'][n],
                                                    num_layers=opts['e_nlayers'][n],
                                                    num_units=opts['e_nfilters'][n],
                                                    output_dim=enc_output_dim,
                                                    scope='encoder/layer_%d' % (n+1),
                                                    reuse=False,
                                                    is_training=self.is_training)
                if opts['encoder'][n] == 'det':
                    encoded = enc_mean
                    if n==opts['nlatents']-1 and opts['prior']=='dirichlet':
                        encoded = tf.nn.softmax(encoded)
                elif opts['encoder'][n] == 'gauss':
                    qz_params = tf.concat((enc_mean,enc_Sigma),axis=-1)
                    encoded = sample_gaussian(opts, qz_params, 'tensorflow')
                else:
                    assert False, 'Unknown encoder %s' % opts['encoder']
                self.encoded.append(encoded)
                Sigma_det = tf.reduce_prod(enc_Sigma,axis=-1)
                Smean, Svar = tf.nn.moments(Sigma_det,axes=[0])
                Sstats = tf.stack([Smean,Svar],axis=-1)
                encSigmas_stats.append(Sstats)
                # - Decoding encoded points (i.e. reconstruct) & reconstruction cost
                if n==0:
                    recon_mean, recon_Sigma = decoder(self.opts, input=encoded,
                                                    archi=opts['d_arch'][n],
                                                    num_layers=opts['d_nlayers'][n],
                                                    num_units=opts['d_nfilters'][n],
                                                    output_dim=2*np.prod(datashapes[opts['dataset']]),
                                                    scope='decoder/layer_%d' % n,
                                                    reuse=False,
                                                    is_training=self.is_training)
                    if opts['decoder'][n] == 'det':
                        reconstructed = recon_mean
                    elif opts['decoder'][n] == 'gauss':
                        p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
                        reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
                    else:
                        assert False, 'Unknown encoder %s' % opts['decoder'][n]
                    if opts['input_normalize_sym']:
                        reconstructed=tf.nn.tanh(reconstructed)
                    else:
                        reconstructed=tf.nn.sigmoid(reconstructed)
                    reconstructed = tf.reshape(reconstructed,[-1]+datashapes[opts['dataset']])
                    loss_reconstruct = reconstruction_loss(opts, self.points,
                                                    reconstructed)
                    self.loss_reconstruct += loss_reconstruct
                else:
                    # Setting output_dim
                    if opts['e_arch'][n-1]=='dcgan' or opts['e_arch'][n-1]=='dcgan_mod':
                        dec_output_dim = 2*datashapes[opts['dataset']][-1]*opts['zdim'][n-1]
                    else:
                        dec_output_dim = 2*opts['zdim'][n-1]
                    recon_mean, recon_Sigma = decoder(self.opts, input=encoded,
                                                    archi=opts['d_arch'][n],
                                                    num_layers=opts['d_nlayers'][n],
                                                    num_units=opts['d_nfilters'][n],
                                                    output_dim=dec_output_dim,
                                                    scope='decoder/layer_%d' % n,
                                                    reuse=False,
                                                    is_training=self.is_training)
                    if opts['decoder'][n] == 'det':
                        reconstructed = recon_mean
                    elif opts['decoder'][n] == 'gauss':
                        p_params = tf.concat((recon_mean,recon_Sigma),axis=-1)
                        reconstructed = sample_gaussian(opts, p_params, 'tensorflow')
                    else:
                        assert False, 'Unknown encoder %s' % opts['decoder'][n]
                    loss_reconstruct = reconstruction_loss(opts, self.encoded[-2],
                                                    reconstructed)
                    self.loss_reconstruct += self.lmbd[n-1] * loss_reconstruct
                self.reconstructed.append(reconstructed)
                self.losses_reconstruct.append(loss_reconstruct)
        self.encSigmas_stats = tf.stack(encSigmas_stats,axis=0)
        # --- Sampling from model (only for generation)
        # reuse variable
        if opts['prior']=='implicit':
            reuse = False
        else:
            reuse = True
        decoded = self.samples
        for n in range(opts['nlatents']-1,-1,-1):
            if n==0:
                decoded_mean, decoded_Sigma = decoder(self.opts, input=decoded,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                output_dim=2*np.prod(datashapes[opts['dataset']]),
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=self.is_training)
                if opts['decoder'][n] == 'det':
                    decoded = decoded_mean
                elif opts['decoder'][n] == 'gauss':
                    p_params = tf.concat((decoded_mean,decoded_Sigma),axis=-1)
                    decoded = sample_gaussian(opts, p_params, 'tensorflow')
                else:
                    assert False, 'Unknown encoder %s' % opts['decoder'][n]
                if opts['input_normalize_sym']:
                    decoded=tf.nn.tanh(decoded)
                else:
                    decoded=tf.nn.sigmoid(decoded)
                decoded = tf.reshape(decoded,[-1]+datashapes[opts['dataset']])
            else:
                # Setting output_dim
                if opts['e_arch'][n-1]=='dcgan' or opts['e_arch'][n-1]=='dcgan_mod':
                    dec_output_dim = 2*datashapes[opts['dataset']][-1]*opts['zdim'][n-1]
                else:
                    dec_output_dim = 2*opts['zdim'][n-1]
                decoded_mean, decoded_Sigma = decoder(self.opts, input=decoded,
                                                archi=opts['d_arch'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                output_dim=dec_output_dim,
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
        # Compute matching penalty cost
        if opts['prior']=='gaussian' or opts['prior']=='dirichlet':
            self.match_penalty = matching_penalty(opts, self.samples, self.encoded[-1])
            self.C = square_dist_v2(self.opts,self.samples, self.encoded[-1])
            imp_match_penalty = matching_penalty(opts, self.decoded[-2], self.encoded[0])
            self.imp_objective = self.losses_reconstruct[0] \
                                + self.lmbd[-1] * imp_match_penalty
        elif opts['prior']=='implicit':
            self.match_penalty = matching_penalty(opts, self.decoded[-2], self.encoded[0])
            self.C = square_dist_v2(self.opts,self.decoded[-2], self.encoded[0])
        else:
            assert False, 'Unknown prior %s' % opts['prior']
        # Compute obj
        self.objective = self.loss_reconstruct \
                         + self.lmbd[-1] * self.match_penalty
        if opts['prior']=='gaussian' or opts['prior']=='dirichlet':
            self.imp_objective = self.losses_reconstruct[0] \
                                + self.lmbd[-1] * imp_match_penalty
        elif opts['prior']=='implicit':
            self.imp_objective = self.objective
        else:
            assert False, 'Unknown prior %s' % opts['prior']

        # Logging info
        self.sinkhorn = sinkhorn_it_v2(self.opts, self.C)
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

    def add_training_placeholders(self):
        opts = self.opts
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        lmbda = tf.placeholder(tf.float32, name='lambda')
        self.lr_decay = decay
        self.is_training = is_training
        self.lmbd = lmbda

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
        if opts['prior']=='gaussian' or opts['prior']=='dirichlet':
            self.pre_loss = moments_loss(self.samples, self.encoded[-1])
        elif opts['prior']=='implicit':
            self.pre_loss = moments_loss(self.decoded[-2], self.encoded[0])

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
        Loss, imp_Loss = [], []
        Loss_rec, Losses_rec, Loss_rec_test = [], [], []
        Loss_match = []
        enc_Sigmas = []
        """
        mean_blurr, fid_scores = [], [],
        """
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
                           self.is_training: True}
                # Update encoder and decoder
                [_, loss, imp_loss, loss_rec, losses_rec, loss_match] = self.sess.run([
                                                self.wae_opt,
                                                self.objective,
                                                self.imp_objective,
                                                self.loss_reconstruct,
                                                self.losses_reconstruct,
                                                self.match_penalty],
                                                feed_dict=feed_dict)
                Loss.append(loss)
                imp_Loss.append(imp_loss)
                Loss_rec.append(loss_rec)
                losses_rec = list(np.array(losses_rec)*np.concatenate((np.ones(1),np.array(wae_lambda[:-1]))))
                Losses_rec.append(losses_rec)
                Loss_match.append(loss_match)
                if opts['vizu_encSigma']:
                    enc_sigmastats = self.sess.run(self.encSigmas_stats,
                                                feed_dict=feed_dict)
                    enc_Sigmas.append(enc_sigmastats)

                ##### TESTING LOOP #####
                if counter % opts['print_every'] == 0:
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
                                                       self.is_training:False})
                        loss_rec_test += l / batches_num_te
                    Loss_rec_test.append(loss_rec_test)


                    # Auto-encoding test images
                    [reconstructed_test, encoded, samples] = self.sess.run(
                                                [self.reconstructed,
                                                 self.encoded,
                                                 self.decoded[:-1]],
                                                feed_dict={self.points:data.test_data[:30*npics],
                                                           self.samples: fixed_noise,
                                                           self.is_training:False})

                    if opts['vizu_embedded'] and counter>1:
                        decoded = samples[::-1]
                        decoded.append(fixed_noise)
                        plot_embedded(opts,encoded,decoded, #[fixed_noise,].append(samples)
                                                data.test_labels[:30*npics],
                                                work_dir,'embedded_e%04d_mb%05d.png' % (epoch, it))
                    if opts['vizu_sinkhorn']:
                        [C,sinkhorn] = self.sess.run([self.C, self.sinkhorn],
                                                    feed_dict={self.points:data.test_data[:npics],
                                                               self.samples: fixed_noise,
                                                               self.is_training:False})
                        plot_sinkhorn(opts, sinkhorn, work_dir,
                                                    'sinkhorn_e%04d_mb%05d.png' % (epoch, it))
                    if opts['vizu_encSigma'] and counter>1:
                        plot_encSigma(opts, enc_Sigmas, work_dir,
                                                    'encSigma_e%04d_mb%05d.png' % (epoch, it))


                    # Auto-encoding training images
                    reconstructed_train = self.sess.run(self.reconstructed,
                                                feed_dict={self.points:data.data[200:200+npics],
                                                           self.is_training:False})
                    # Random samples generated by the model
                    samples = self.sess.run(self.decoded,
                                                feed_dict={self.points:data.data[200:200+npics],
                                                           self.samples: fixed_noise,
                                                           self.is_training: False})

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
                    if opts['vizu_sinkhorn']:
                        debug_str = 'mdist=%10.3e, Mdist=%10.3e, avgdist=%10.3e\n ' % (
                                                    np.amin(C),
                                                    np.amax(C),
                                                    np.mean(C))
                        logging.error(debug_str)

                    """
                    debug_str = 'FID=%.3f, BLUR=%10.4e' % (
                                                fid_scores[-1],
                                                mean_blurr[-1])
                    logging.error(debug_str)
                    """

                    print('')
                    # Making plots
                    if opts['prior']=='gaussian' or opts['prior']=='dirichlet':
                        samples_prior = fixed_noise
                    elif opts['prior']=='implicit':
                        samples_prior = samples[-2]
                    save_train(opts, data.data[200:200+npics], data.test_data[:npics],  # images
                                     data.test_labels[:30*npics],    # labels
                                     reconstructed_train[0], reconstructed_test[0][:npics], # reconstructions
                                     encoded[-1],   # encoded points (bottom)
                                     samples_prior, samples[-1],  # prior samples, model samples
                                     Loss, imp_Loss, Loss_match,  # losses
                                     Loss_rec, Loss_rec_test,   # rec losses
                                     Losses_rec,    # rec losses for each latents
                                     work_dir,  # working directory
                                     'res_e%04d_mb%05d.png' % (epoch, it))  # filename

                # Update learning rate if necessary and counter
                # First 20 epochs do nothing
                if epoch >= 1000:
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
                    if epoch >= 800 and len(Loss_rec) > 0:
                        # if np.mean(Loss[-10:]) < np.mean(Loss[-10 * batches_num:])-1.*np.var(Loss[-10 * batches_num:]):
                        #     wait_lambda = 0
                        # else:
                        #     wait_lambda += 1
                        if wait_lambda > 200 * batches_num:
                            opts['lambda_scalar'] *= 1.6
                            opts['lambda'] = [opts['lambda_scalar']**(1/i) for i in range(opts['nlatents'],0,-1)]
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
        if epoch > 0:
            self.saver.save(self.sess, os.path.join(work_dir,
                                                'checkpoints',
                                                'trained-wae-final'),
                                                global_step=counter)


    def latent_interpolation(self, data, MODEL_PATH, WEIGHTS_FILE):
        """
        Plot and save different latent interpolation
        """

        opts = self.opts
        # Load trained weights
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)
        # Set up
        test_size = np.shape(data.test_data)[0]
        num_steps = 40
        num_anchors = 10
        imshape = datashapes[opts['dataset']]

        # Auto-encoding test images
        logging.error('Encoding test images..')
        num_pics = 5000
        encoded = self.sess.run(self.encoded,
                                feed_dict={self.points:data.test_data[:num_pics],
                                           self.is_training:False})

        # Encode anchors points and interpolate
        if opts['prior']!='implicit':
            logging.error('Anchors interpolation..')
            anchors_ids = np.random.choice(num_pics,2*num_anchors,replace=False)
            data_anchors = data.test_data[anchors_ids]
            enc_anchors = encoded[-1][anchors_ids]
            enc_interpolation = linespace(opts, num_steps,
                                    anchors=np.reshape(enc_anchors,[-1,2]+encshape))
            dec_anchors = self.sess.run(self.decoded[-1],
                                    feed_dict={self.samples: np.reshape(enc_interpolation,[-1,]+encshape),
                                               self.is_training: False})
            inter_anchors = np.reshape(dec_anchors,[-1,num_steps]+imshape)
        else:
            encshape = list([opts['zdim'][-1]])
            inter_anchors = None
        # Latent interpolation
        logging.error('Latent interpolation..')
        if opts['prior']!='implicit':
            enc_mean = np.mean(encoded[-1],axis=0)
            enc_var = np.mean(np.square(encoded[-1]-enc_mean),axis=0)
        else:
            enc_mean = np.zeros(opts['zdim'][-1], dtype='float32')
            enc_var = np.ones(opts['zdim'][-1], dtype='float32')
        mins, maxs = enc_mean - 3*np.sqrt(enc_var), enc_mean + 3*np.sqrt(enc_var)
        x = np.linspace(mins[0], maxs[0], num=num_steps, endpoint=True)
        xymin = np.stack([x,mins[1]*np.ones(num_steps)],axis=-1)
        xymax = np.stack([x,maxs[1]*np.ones(num_steps)],axis=-1)
        latent_anchors = np.stack([xymin,xymax],axis=1)
        grid_interpolation = linespace(opts, num_steps,
                                anchors=latent_anchors)
        dec_latent = self.sess.run(self.decoded[-1],
                                feed_dict={self.samples: np.reshape(grid_interpolation,[-1,]+encshape),
                                           self.is_training: False})
        inter_latent = np.reshape(dec_latent,[-1,num_steps]+imshape)
        # Samples generation
        logging.error('Samples generation..')
        num_cols = 15
        npics = num_cols**2
        prior_noise = sample_pz(opts, self.pz_params, npics)
        samples = self.sess.run(self.decoded[-1],
                               feed_dict={self.samples: prior_noise,
                                          self.is_training: False})
        # Making plots
        logging.error('Saving images..')
        save_latent_interpolation(opts, data.test_labels[:num_pics], # labels
                        encoded, # encoded points
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
