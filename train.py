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
from sampling_functions import sample_pz, sample_gaussian, sample_unif, linespace
from models import WAE, stackedWAE, VAE
from plot_functions import save_train, save_latent_interpolation, save_vlae_experiment
from plot_functions import plot_splitloss, plot_samples, plot_fullrec, plot_embedded, plot_latent, plot_grid, plot_stochasticity
# from plot_functions import plot_splitloss, plot_fullrec, plot_embedded, plot_latent, plot_grid, plot_stochasticity

# Path to inception model and stats for training set
sys.path.append('../TTUR')
sys.path.append('../inception')
import fid
inception_path = '../inception'
inception_model = os.path.join(inception_path, 'classify_image_graph_def.pb')
layername = 'FID_Inception_Net/pool_3:0'

import pdb

class Run(object):

    def __init__(self, opts, data):

        logging.error('Building the Tensorflow Graph')
        self.opts = opts

        # --- Data shape
        self.data = data

        # --- Placeholders
        self.add_placeholders()

        # --- Initialize prior parameters
        mean = np.zeros(opts['zdim'][-1], dtype='float32')
        Sigma = np.ones(opts['zdim'][-1], dtype='float32')
        self.pz_params = np.concatenate([mean, Sigma], axis=-1)

        # --- Instantiate Model
        if self.opts['model'] == 'vae':
            self.model = VAE(self.opts, self.pz_params)
        elif self.opts['model'] == 'wae':
            self.model = WAE(self.opts, self.pz_params)
        elif self.opts['model'] == 'stackedwae':
            self.model = stackedWAE(self.opts, self.pz_params)
        elif self.opts['model'] == 'LVAE':
            raise NotImplementedError()
        else:
            raise Exception('Unknown {} model'.format(self.opts['model']))

        # --- Sample next batch
        x = self.data.next_element

        # --- Objective
        losses = self.model.losses(x, self.sigma_scale, False, self.is_training)
        self.obs_cost, self.latent_costs, self.matching_penalty = losses[0], losses[1], losses[2]
        self.enc_Sigma_penalty, self.dec_Sigma_penalty = losses[3], losses[4]
        # Compute obj
        latent_loss = 0.
        for i in range(len(self.latent_costs)):
            latent_loss += self.latent_costs[i]*self.lmbd[i]
        self.objective = self.obs_cost + latent_loss + self.lmbd[-1] * self.matching_penalty
        # Enc Sigma penalty
        if self.opts['enc_sigma_pen']:
            Sigma_pen = 0.
            for i in range(len(self.enc_Sigma_penalty)):
                Sigma_pen += self.enc_Sigma_penalty[i]*self.lmbd_sigma[i]
            self.objective -= Sigma_pen
        # Dec Sigma penalty
        if self.opts['dec_sigma_pen']:
            Sigma_pen = 0.
            for i in range(len(self.dec_Sigma_penalty)):
                Sigma_pen += self.dec_Sigma_penalty[i]*self.lmbd_sigma[i]
            self.objective -= Sigma_pen

        # --- Various metrics
        # MSE
        self.mse = self.model.MSE(x, self.sigma_scale)
        # blurriness
        self.blurriness = self.model.blurriness(x)
        # real imagesnblurriness
        self.real_blurriness = self.model.blurriness(self.images)
        # layerwise KL
        if self.opts['model'] == 'wae':
            self.KL = [tf.ones([]) for i in range(self.opts['nlatents'])]
        else:
            self.KL = self.model.layerwise_kl(x, self.sigma_scale)
        # FID
        if opts['fid']:
            self.inception_graph = tf.Graph()
            self.inception_sess = tf.compat.v1.Session(graph=self.inception_graph)
            self.inception_layer = self.model.inception_Net(self.inception_sess, self.inception_graph)

        # --- encode and reconstruct
        # encode
        self.encoded, _, _ = self.model.encode(self.images, self.sigma_scale,
                                    False, reuse=True, is_training=False)
        # reconstruct
        reconstruction = self.model.reconstruct(self.encoded, self.sigma_scale)
        shape = [-1,] + self.data.data_shape
        self.reconstruction = [tf.reshape(reconstruction[n], shape) for n in range(len(reconstruction))]

        # --- Sampling from model (only for generation)
        decoded, _, _ = self.model.sample_x_from_prior(self.pz_samples, self.sigma_scale)
        self.samples = tf.reshape(decoded[-1], [-1,]+self.data.data_shape)

        # --- encode multi samples and reconstruct
        # encode
        encoded, _, _ = self.model.encode(self.images, self.sigma_scale,
                                    True, self.opts['nresamples'], True, is_training=False)
        # reconstruct
        reconstruction = self.model.reconstruct(encoded, tf.ones(self.sigma_scale.get_shape()))
        shape = [-1, self.model.opts['nresamples']] + self.data.data_shape
        self.resample_reconstruction = [tf.reshape(reconstruction[n], shape) for n in range(len(reconstruction))]

        # # --- Point interpolation for implicit-prior WAE (only for generation)
        # self.anchor_interpolation = self.anchor_interpolate()

        # --- Optimizers
        self.add_optimizers()

        # --- Init iteratorssess, saver and load trained weights if needed, else init variables
        self.sess = tf.compat.v1.Session()
        self.train_handle, self.test_handle = self.data.init_iterator(self.sess)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=10)
        self.initializer = tf.compat.v1.global_variables_initializer()
        self.sess.graph.finalize()

    def add_placeholders(self):
        # inputs ph
        self.images = tf.compat.v1.placeholder(tf.float32, [None,] + self.data.data_shape,
                                    name='images_ph')
        self.pz_samples = tf.compat.v1.placeholder(tf.float32, [None,] + [self.opts['zdim'][-1],],
                                    name='pzsamples_ph')
        # obj coef ph
        self.lmbd = tf.compat.v1.placeholder(tf.float32, [self.opts['nlatents'],],
                                    name='lambda_ph')
        self.lmbd_sigma = tf.compat.v1.placeholder(tf.float32, [self.opts['nlatents'],],
                                    name='lambda_sigma_ph')
        # training parms ph
        self.lr_decay = tf.compat.v1.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training_ph')
        # vizu config ph
        self.sigma_scale = tf.compat.v1.placeholder(tf.float32, [1,], name='sigma_scale_ph')
        if self.opts['archi'][0]=='resnet_v2' and self.opts['nlatents']!=1:
            if self.opts['e_resample'][0]=='down':
                self.anchors_points = tf.compat.v1.placeholder(tf.float32,
                                    [None] + [int(self.data_shape[0]/2)*int(self.data_shape[1]/2)*self.opts['zdim'][0],],
                                    name='anchors_ph')
            else:
                self.anchors_points = tf.compat.v1.placeholder(tf.float32,
                                    [None] + [self.data_shape[0]*self.data_shape[1]*self.opts['zdim'][0],],
                                    name='anchors_ph')
        else:
            self.anchors_points = tf.compat.v1.placeholder(tf.float32,
                                    [None] + [self.opts['zdim'][0],],
                                    name='anchors_ph')

    # def anchor_interpolate(self):
    #     # Anchor interpolation for 1-layer encoder
    #     opts = self.opts
    #     if opts['d_archi'][0]=='resnet_v2':
    #         features_dim=self.features_dim[1]
    #     else:
    #         features_dim=self.features_dim[0]
    #     output_dim = datashapes[opts['dataset']][:-1]+[2*datashapes[opts['dataset']][-1],]
    #     anc_mean, anc_Sigma = decoder(self.opts, input=self.anchors_points,
    #                                             archi=opts['d_archi'][0],
    #                                             num_layers=opts['d_nlayers'][0],
    #                                             num_units=opts['d_nfilters'][0],
    #                                             filter_size=opts['filter_size'][0],
    #                                             output_dim=output_dim,
    #                                             features_dim=features_dim,
    #                                             resample=opts['d_resample'][0],
    #                                             last_archi=opts['d_last_archi'][0],
    #                                             scope='decoder/layer_0',
    #                                             reuse=True,
    #                                             is_training=False)
    #     if opts['decoder'][0] == 'det':
    #         anchors_decoded = anc_mean
    #     elif opts['decoder'][0] == 'gauss':
    #         p_params = tf.concat((anc_mean,anc_Sigma),axis=-1)
    #         anchors_decoded = sample_gaussian(opts, p_params, 'tensorflow')
    #     elif opts['decoder'][0] == 'bernoulli':
    #         anchors_decoded = sample_bernoulli(anc_mean)
    #     else:
    #         assert False, 'Unknown encoder %s' % opts['decoder'][0]
    #     if opts['decoder'][0] != 'bernoulli':
    #         if opts['input_normalize_sym']:
    #             anchors_decoded=tf.nn.tanh(anchors_decoded)
    #         else:
    #             anchors_decoded=tf.nn.sigmoid(anchors_decoded)
    #     self.anchors_decoded = tf.reshape(anchors_decoded,[-1]+datashapes[opts['dataset']])

    def add_savers(self):
        opts = self.opts
        saver = tf.compat.v1.train.Saver(max_to_keep=10)
        self.saver = saver

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if opts['optimizer'] == 'sgd':
            return tf.train.compat.v1.train.GradientDescentOptimizer(lr)
        elif opts['optimizer'] == 'adam':
            return tf.compat.v1.train.AdamOptimizer(lr, beta1=opts['adam_beta1'], beta2=opts['adam_beta2'])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        opts = self.opts
        # WAE optimizer
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder')
        decoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='decoder')
        ae_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = opt.minimize(loss=self.objective, var_list=ae_vars)
        # Pretraining optimizer
        if opts['pretrain']:
            pre_opt = self.optimizer(0.001)
            self.pre_opt = pre_opt.minimize(loss=self.pre_loss, var_list=encoder_vars)


    def train(self, WEIGHTS_PATH=None):
        """
        Train top-down model with chosen method
        """

        logging.error('\nTraining  {} with {} latent layers\n'.format(self.opts['model'], self.opts['nlatents']))
        exp_dir = self.opts['exp_dir']

        # - Set up for training
        logging.error('\nTrain size: {}, Batch num.: {}, Ite. num: {}'.format(
                                    self.data.train_size,
                                    int(self.data.train_size/self.opts['batch_size']),
                                    self.opts['it_num']))
        npics = self.opts['plot_num_pics']
        fixed_noise = sample_pz(self.opts, self.pz_params, npics)

        # Compute blurriness of real data
        idx = np.random.randint(0, high=self.data.test_size, size=1000)
        images = self.data.data_test[idx]
        real_blurr = self.sess.run(self.real_blurriness, feed_dict={self.images: images})
        logging.error('Real pictures sharpness = %10.4e' % np.min(real_blurr))
        print('')


        # - FID
        if self.opts['fid']:
            # Load inception mean samples for train set
            trained_stats = os.path.join(inception_path, 'fid_stats.npz')
            # Load trained stats
            f = np.load(trained_stats)
            self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
            f.close()

        # Init all monitoring variables
        trLoss, trLoss_obs, trLoss_latent, trLoss_match = [], [], [], []
        trenc_Sigma_reg, trdec_Sigma_reg = [], []
        trMSE, trBlurr, trKL = [], [], []
        teLoss, teLoss_obs, teLoss_latent, teLoss_match = [], [], [], []
        teenc_Sigma_reg, tedec_Sigma_reg = [], []
        teMSE, teBlurr, teKL = [], [], []
        FID = []
        # lr schedule
        decay, decay_warmup, decay_steps, decay_rate = 1., 10000, 10000, 0.99
        # lambda schedule
        annealed_warmup = 50000
        # lreg_init = self.opts['lambda_reg_init']
        # lreg_final = self.opts['lambda'][-1]
        # lmbd = self.opts['lambda'][:-1].append(lreg_init)
        if self.opts['lambda_schedule'] == 'adaptive':
            lmbd = self.opts['lambda_init']
        else:
            lmbd = self.opts['lambda']
        lmbd_sigma = self.opts['lambda_sigma']

        # - Init sess and load trained weights if needed
        if self.opts['use_trained']:
            if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_PATH)
        else:
            self.sess.run(self.initializer)
            # if self.opts['pretrain']:
            #     logging.error('Pretraining the encoder\n')
            #     self.pretrain_encoder(data)
            #     print('')

        ##### TRAINING LOOP #####
        for it in range(self.opts['it_num']):
            # Saver
            if it > 0 and it % self.opts['save_every'] == 0:
                self.saver.save(self.sess, os.path.join(
                                    exp_dir,
                                    'checkpoints',
                                    'trained-wae'),
                                    global_step=it)

            # optimization step
            it += 1
            _ = self.sess.run(self.opt, feed_dict={self.data.handle: self.train_handle,
                                    self.sigma_scale: np.ones(1),
                                    self.lmbd: lmbd,
                                    self.lmbd_sigma: lmbd_sigma,
                                    self.lr_decay: decay,
                                    self.is_training: True})


            ##### TESTING LOOP #####
            if it % self.opts['evaluate_every'] == 0:
                logging.error('\nIteration {}/{}'.format(it, self.opts['it_num']))
                # training losses
                [loss, obs_cost, latent_costs, matching_penalty, enc_Sigma_penalty, dec_Sigma_penalty] = self.sess.run(
                                    [self.objective,
                                    self.obs_cost,
                                    self.latent_costs,
                                    self.matching_penalty,
                                    self.enc_Sigma_penalty,
                                    self.dec_Sigma_penalty],
                                    feed_dict={self.data.handle: self.train_handle,
                                                self.sigma_scale: np.ones(1),
                                                self.lmbd: lmbd,
                                                self.lmbd_sigma: lmbd_sigma,
                                                self.is_training: False})
                trLoss.append(loss)
                trLoss_obs.append(obs_cost)
                trLoss_latent.append([lmbd[n]*latent_costs[n] for n in range(len(latent_costs))])
                trLoss_match.append(lmbd[-1]*matching_penalty)
                trenc_Sigma_reg.append([lmbd_sigma[n]*enc_Sigma_penalty[n] for n in range(len(enc_Sigma_penalty))])
                trdec_Sigma_reg.append([lmbd_sigma[n]*dec_Sigma_penalty[n] for n in range(len(dec_Sigma_penalty))])
                # training metrics
                idx = np.random.randint(0, self.data.train_size, self.opts['batch_size'])
                batch, _ = self.data.sample_observations(idx, 'train')
                [mse, blurr, kl] = self.sess.run([self.mse, self.blurriness, self.KL],
                                    feed_dict={self.data.handle: self.train_handle,
                                                self.images: batch,
                                                self.sigma_scale: np.ones(1),})
                trMSE.append(mse)
                trBlurr.append(blurr)
                trKL.append(kl)
                # init testing losses & metrics
                test_it_num = int(self.data.test_size / self.opts['batch_size'])
                loss, obs_cost, matching_penalty = 0., 0., 0.
                enc_Sigma_penalty, dec_Sigma_penalty = np.zeros(len(enc_Sigma_penalty)), np.zeros(len(dec_Sigma_penalty))
                latent_costs = np.zeros(len(latent_costs))
                mse, blurr, kl = 0., 0., np.zeros(len(kl))
                for it_ in range(test_it_num):
                    # testing losses
                    [l, obs, latent, match, eSigma, dSigma] = self.sess.run([self.objective,
                                    self.obs_cost,
                                    self.latent_costs,
                                    self.matching_penalty,
                                    self.enc_Sigma_penalty,
                                    self.dec_Sigma_penalty],
                                    feed_dict={self.data.handle: self.test_handle,
                                                self.sigma_scale: np.ones(1),
                                                self.lmbd: lmbd,
                                                self.lmbd_sigma: lmbd_sigma,
                                                self.is_training: False})
                    loss += l / test_it_num
                    obs_cost += obs / test_it_num
                    matching_penalty += match / test_it_num
                    enc_Sigma_penalty += np.array(eSigma) / test_it_num
                    dec_Sigma_penalty += np.array(dSigma) / test_it_num
                    latent_costs += np.array(latent) / test_it_num
                    # testing metrics
                    idx = np.random.randint(0, self.data.test_size, self.opts['batch_size'])
                    batch, _ = self.data.sample_observations(idx)
                    [m, b, k] = self.sess.run([self.mse, self.blurriness, self.KL],
                                    feed_dict={self.data.handle: self.test_handle,
                                                self.images: batch,
                                                self.sigma_scale: np.ones(1)})
                    mse += m / test_it_num
                    blurr += b / test_it_num
                    kl += np.array(k) / test_it_num
                teLoss.append(loss)
                teLoss_obs.append(obs_cost)
                teLoss_latent.append([lmbd[n]*latent_costs[n] for n in range(len(latent_costs))])
                teLoss_match.append(lmbd[-1]*matching_penalty)
                teenc_Sigma_reg.append([lmbd_sigma[n]*enc_Sigma_penalty[n] for n in range(len(enc_Sigma_penalty))])
                tedec_Sigma_reg.append([lmbd_sigma[n]*dec_Sigma_penalty[n] for n in range(len(dec_Sigma_penalty))])
                teMSE.append(mse)
                teBlurr.append(blurr)
                teKL.append(kl)
                # log output
                debug_str = 'ITER: %d/%d, ' % (it, self.opts['it_num'])
                logging.error(debug_str)
                debug_str = 'teLOSS=%.3f, trLOSS=%.3f' % (teLoss[-1], trLoss[-1])
                logging.error(debug_str)
                debug_str = 'REC=%.3f, LATENT=%.3f, MATCH=%10.3e, eSIGMA=%10.3e, dSIGMA=%10.3e'  % (
                                    teLoss_obs[-1],
                                    np.sum(teLoss_latent[-1]),
                                    teLoss_match[-1],
                                    np.sum(teenc_Sigma_reg[-1]),
                                    np.sum(tedec_Sigma_reg[-1]))
                logging.error(debug_str)
                debug_str = 'MSE=%.3f, BLURR=%.3f, KL=%10.3e\n '  % (
                                    teMSE[-1],
                                    teBlurr[-1],
                                    np.mean(teKL[-1]/self.opts['zdim']))
                logging.error(debug_str)

                # Compute FID score
                if self.opts['fid']:
                    fid = 0
                    for it_ in range(test_it_num):
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


            ##### Vizu #####
            if it % self.opts['print_every'] == 0:
                np.random.seed(1234)
                # Auto-encoding test images & samples generated by the model
                idx = np.random.randint(0, self.data.test_size, npics)
                batch, labels = self.data.sample_observations(idx)
                [rec, encoded, samples] = self.sess.run([self.reconstruction,
                                    self.encoded[-1],
                                    self.samples],
                                    feed_dict={self.images: batch,
                                               self.pz_samples: fixed_noise,
                                                self.sigma_scale: np.ones(1)})
                # rec = [rec[n][:,0] for n in range(len(rec))]
                # encoded = encoded[:,0]
                save_train(self.opts, batch, labels, rec[-1], samples,
                                    encoded, fixed_noise,
                                    teLoss, teLoss_obs,
                                    teLoss_latent, teLoss_match,
                                    teenc_Sigma_reg, tedec_Sigma_reg,
                                    trLoss, trLoss_obs,
                                    trLoss_latent, trLoss_match,
                                    teMSE, teBlurr, teKL,
                                    trMSE, trBlurr, trKL,
                                    exp_dir, 'train_plots', 'res_it%07d.png' % it)

                if self.opts['vizu_splitloss']:
                    plot_splitloss(self.opts, teLoss_obs, teLoss_latent, teLoss_match,
                                    teenc_Sigma_reg, tedec_Sigma_reg,
                                    exp_dir, 'train_plots', 'losses_it%07d.png' % it)

                if self.opts['vizu_fullrec']:
                    plot_fullrec(self.opts, batch[:10], [rec[n][:10] for n in range(len(rec))], exp_dir, 'train_plots', 'rec_it%07d.png' % it)

                if self.opts['vizu_embedded']:
                    batchsize = 1000
                    # zs, ys = [], []
                    # for _ in range(int(nencoded/batchsize)):
                    idx = np.random.randint(0, self.data.test_size, batchsize)
                    x, y = self.data.sample_observations(idx)
                    z = self.sess.run(self.encoded, feed_dict={self.images: x,
                                    self.sigma_scale: np.ones(1)})
                        # zs += [z[n][:,0] for n in range(len(z))]
                        # ys += y
                    plot_embedded(self.opts, z, y, exp_dir, 'train_plots', 'emb_it%07d.png' % it)

                if self.opts['vizu_latent']:
                    # idx = np.random.randint(0, self.data.test_size, int(sqrt(self.opts['nresamples'])))
                    idx = np.random.randint(0, self.data.test_size, 4)
                    x, _ = self.data.sample_observations(idx)
                    reconstruction = self.sess.run(self.resample_reconstruction, feed_dict={
                                    self.images: x,
                                    self.sigma_scale: self.opts['sigma_scale_resample']})
                    plot_latent(self.opts, reconstruction, exp_dir, 'train_plots', 'latent_expl_it%07d.png' % it)

                if self.opts['vizu_pz_grid']:
                    num_cols = 10
                    enc_mean = np.zeros(self.opts['zdim'][-1], dtype='float32')
                    enc_var = np.ones(self.opts['zdim'][-1], dtype='float32')
                    mins, maxs = enc_mean - 2.*np.sqrt(enc_var), enc_mean + 2.*np.sqrt(enc_var)
                    x = np.linspace(mins[0], maxs[0], num=num_cols, endpoint=True)
                    xymin = np.stack([x,mins[1]*np.ones(num_cols)],axis=-1)
                    xymax = np.stack([x,maxs[1]*np.ones(num_cols)],axis=-1)
                    anchors = np.stack([xymin,xymax],axis=1)
                    grid = linespace(self.opts, num_cols, anchors=anchors)
                    grid = grid.reshape([-1,self.opts['zdim'][-1]])
                    samples = self.sess.run(self.samples, feed_dict={
                                    self.pz_samples: grid,
                                    self.sigma_scale: np.ones(1)})
                    samples = samples.reshape([-1,num_cols]+self.data.data_shape)
                    plot_grid(self.opts, samples, exp_dir, 'train_plots', 'pz_grid_it%07d.png' % it)

                if self.opts['vizu_stochasticity']:
                    Samples = []
                    for n in range(len(self.opts['sigma_scale_stochasticity'])):
                        samples = self.sess.run(self.samples, feed_dict={
                                        self.pz_samples: fixed_noise[:5],
                                        self.sigma_scale: self.opts['sigma_scale_stochasticity'][n]})
                        Samples.append(samples)
                    plot_stochasticity(self.opts, Samples, exp_dir, 'train_plots', 'stochasticity_it%07d.png' % it)

                np.random.seed()


            ##### lr #####
            #Update learning rate if necessary and counter
            # First 150 epochs do nothing
            if it >= decay_warmup and it % decay_steps == 0:
                decay = decay_rate ** (int(it / decay_steps))
                logging.error('Reduction in lr: %f\n' % decay)
                """
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
                """

            ##### lambda #####
            # Update regularizer if necessary
            if self.opts['lambda_schedule'] == 'adaptive':
                lmbd = [min(self.opts['lambda'][n], self.opts['lambda_init'][n]+it*(self.opts['lambda'][n]-self.opts['lambda_init'][n])/annealed_warmup) for n in range(len(lmbd))]
                if it%2000==0:
                    debug_str = 'Lambda update: l1=%10.3e, l2=%10.3e, l3=%10.3e, l4=%10.3e, l5=%10.3e\n'  % (
                                    lmbd[0], lmbd[1], lmbd[2], lmbd[3], lmbd[4])
                    logging.error(debug_str)

                # if it > 1 and len(teLoss) > 0:
                #     if wait_lambda > 50000 + 1:
                #         # opts['lambda'] = list(2*np.array(opts['lambda']))
                #         self.opts['lambda'][-1] = 2*self.opts['lambda'][-1]
                #         wae_lambda = self.opts['lambda']
                #         logging.error('Lambda updated to %s\n' % wae_lambda)
                #         print('')
                #         wait_lambda = 0
                #     else:
                #         wait_lambda+=1

        ##### saving #####
        # Save the final model
        if self.opts['save_final']:
            self.saver.save(self.sess, os.path.join(exp_dir,
                                                'checkpoints',
                                                'trained-wae-final'),
                                                global_step=it)
        # save training data
        if self.opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(exp_dir, data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path, name), teLoss=np.array(teLoss),
                        teLoss_obs=np.array(teLoss_obs),
                        teLoss_match=np.array(teLoss_match),
                        teenc_Sigma_reg=np.array(teenc_Sigma_reg),
                        tedec_Sigma_reg=np.array(tedec_Sigma_reg),
                        trLoss=np.array(trLoss), trLoss_obs=np.array(trLoss_obs),
                        trLoss_match=np.array(trLoss_match), teMSE=np.array(teMSE),
                        teBlurr=np.array(teBlurr), teKL=np.array(teKL),
                        trMSE=np.array(trMSE), trBlurr=np.array(trBlurr), trKL=np.array(trKL))

    def test(self, WEIGHTS_PATH=None):
        """
        Test and plot
        """

        logging.error('\nTraining  {} with {} latent layers\n'.format(self.opts['model'], self.opts['nlatents']))
        exp_dir = self.opts['exp_dir']

        # Init model hyper params
        npics = self.opts['plot_num_pics']
        fixed_noise = sample_pz(self.opts, self.pz_params, npics)
        lmbd = self.opts['lambda']
        lmbd_sigma = self.opts['lambda_sigma']

        # Compute blurriness of real data
        idx = np.random.randint(0, high=self.data.test_size, size=1000)
        images = self.data.data_test[idx]
        real_blurr = self.sess.run(self.real_blurriness, feed_dict={self.images: images})
        logging.error('Real pictures sharpness = %10.4e' % np.min(real_blurr))
        print('')


        # - Init sess and load trained weights if needed
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)
        ##### TESTING LOOP #####
        # Init all monitoring variables
        trLoss, trLoss_obs, trLoss_match = 0., 0., 0.
        teLoss, teLoss_obs, teLoss_match = 0., 0., 0.
        trLoss_latent = np.zeros(len(self.opts['nlatents'])-1)
        teLoss_latent = np.zeros(len(self.opts['nlatents'])-1)
        trMSE, trBlurr, teMSE, teBlurr = 0., 0., 0., 0.
        trKL = np.zeros(len(self.opts['nlatents']))
        teKL = np.zeros(len(self.opts['nlatents']))

        it_num = int(self.data.test_size / self.opts['batch_size'])
        for it in range(test_it_num):
            # training losses
            [l, obs, latent, match] = self.sess.run(
                                [self.objective,
                                self.obs_cost,
                                self.latent_costs,
                                self.matching_penalty],
                                feed_dict={self.data.handle: self.train_handle,
                                            self.sigma_scale: np.ones(1),
                                            self.lmbd: lmbd,
                                            self.lmbd_sigma: lmbd_sigma,
                                            self.is_training: False})
            trLoss += l / it_num
            trLoss_obs += obs / it_num
            trLoss_match += match / it_num
            trLoss_latent += np.array(latent) / it_num
            # training metrics
            idx = np.random.randint(0, self.data.train_size, self.opts['batch_size'])
            batch, _ = self.data.sample_observations(idx, 'train')
            [mse, blurr, kl] = self.sess.run([self.mse, self.blurriness, self.KL],
                                feed_dict={self.data.handle: self.train_handle,
                                            self.images: batch,
                                            self.sigma_scale: np.ones(1),})
            trMSE += mse / it_num
            trBlurr += blurr / it_num
            trKL += np.array(kl) / it_num
            # testing losses
            [l, obs, latent, match] = self.sess.run([self.objective,
                            self.obs_cost,
                            self.latent_costs,
                            self.matching_penalty],
                            feed_dict={self.data.handle: self.test_handle,
                                        self.sigma_scale: np.ones(1),
                                        self.lmbd: lmbd,
                                        self.lmbd_sigma: lmbd_sigma,
                                        self.is_training: False})
            teLoss += l / it_num
            teLoss_obs += obs / it_num
            teLoss_match += match / it_num
            teLoss_latent += np.array(latent) / it_num
            # testing metrics
            idx = np.random.randint(0, self.data.test_size, self.opts['batch_size'])
            batch, _ = self.data.sample_observations(idx)
            [mse, blurr, kl] = self.sess.run([self.mse, self.blurriness, self.KL],
                            feed_dict={self.data.handle: self.test_handle,
                                        self.images: batch,
                                        self.sigma_scale: np.ones(1)})
            teMSE += mse / it_num
            teBlurr += blurr / it_num
            teKL += np.array(kl) / it_num
        trLoss_latent *= lmbd
        teLoss_latent *= lmbd
        # logging output
        debug_str = 'teLOSS=%.3f, trLOSS=%.3f' % (teLoss, trLoss)
        logging.error(debug_str)
        debug_str = 'REC=%.3f, LATENT=%.3f, MATCH=%10.3e'  % (
                            teLoss_obs,
                            np.sum(teLoss_latent),
                            teLoss_match)
        logging.error(debug_str)
        debug_str = 'MSE=%.3f, BLURR=%.3f, KL=%10.3e\n '  % (
                            teMSE,
                            teBlurr,
                            np.mean(teKL/self.opts['zdim']))
        logging.error(debug_str)
        # save test data
        data_dir = 'test_data'
        save_path = os.path.join(exp_dir, data_dir)
        utils.create_dir(save_path)
        name = 'res_final'
        np.savez(os.path.join(save_path, name), teLoss=np.array(teLoss),
                    teLoss_obs=np.array(teLoss_obs),
                    teLoss_match=np.array(teLoss_match),
                    trLoss=np.array(trLoss), trLoss_obs=np.array(trLoss_obs),
                    trLoss_match=np.array(trLoss_match), teMSE=np.array(teMSE),
                    teBlurr=np.array(teBlurr), teKL=np.array(teKL),
                    trMSE=np.array(trMSE), trBlurr=np.array(trBlurr), trKL=np.array(trKL))

        ##### Vizu #####
        if self.opts['vizu_samples']:
            samples = self.sess.run(self.samples, feed_dict={
                            self.pz_samples: fixed_noise,
                            self.sigma_scale: np.ones(1)})
            plot_samples(self.opts, samples, exp_dir, 'test_plots', 'samples.png')

        if self.opts['vizu_fullrec']:
            # idx = np.random.randint(0, self.data.test_size, npics)
            idx = [21, 7, 24, 18, 28, 12, 75, 82, 32]
            batch, _ = self.data.sample_observations(idx)
            rec = self.sess.run(self.reconstruction, feed_dict={
                            self.images: batch,
                            self.sigma_scale: np.ones(1)})
            plot_fullrec(self.opts, batch, rec, exp_dir, 'test_plots', 'rec.png')

        if self.opts['vizu_embedded']:
            batchsize = 5000
            # zs, ys = [], []
            # for _ in range(int(nencoded/batchsize)):
            idx = np.random.randint(0, self.data.test_size, batchsize)
            x, y = self.data.sample_observations(idx)
            z = self.sess.run(self.encoded, feed_dict={self.images: x,
                            self.sigma_scale: np.ones(1)})
            plot_embedded(self.opts, z, y, exp_dir, 'test_plots', 'emb.png' % it)

        if self.opts['vizu_latent']:
            np.random.seed(1234)
            idx = np.random.randint(0, self.data.test_size, 4)
            x, _ = self.data.sample_observations(idx)
            reconstruction = self.sess.run(self.resample_reconstruction, feed_dict={
                            self.images: x,
                            self.sigma_scale: self.opts['sigma_scale_resample']})
            plot_latent(self.opts, reconstruction, exp_dir, 'test_plots', 'latent_expl.png' % it)
            np.random.seed()

        if self.opts['vizu_pz_grid']:
            num_cols = 10
            enc_mean = np.zeros(self.opts['zdim'][-1], dtype='float32')
            enc_var = np.ones(self.opts['zdim'][-1], dtype='float32')
            mins, maxs = enc_mean - 2.*np.sqrt(enc_var), enc_mean + 2.*np.sqrt(enc_var)
            x = np.linspace(mins[0], maxs[0], num=num_cols, endpoint=True)
            xymin = np.stack([x,mins[1]*np.ones(num_cols)],axis=-1)
            xymax = np.stack([x,maxs[1]*np.ones(num_cols)],axis=-1)
            anchors = np.stack([xymin,xymax],axis=1)
            grid = linespace(self.opts, num_cols, anchors=anchors)
            grid = grid.reshape([-1,self.opts['zdim'][-1]])
            samples = self.sess.run(self.samples, feed_dict={
                            self.pz_samples: grid,
                            self.sigma_scale: np.ones(1)})
            samples = samples.reshape([-1,num_cols]+self.data.data_shape)
            plot_grid(self.opts, samples, exp_dir, 'test_plots', 'pz_grid.png' % it)

        if self.opts['vizu_stochasticity']:
            Samples = []
            for n in range(len(self.opts['sigma_scale_stochasticity'])):
                samples = self.sess.run(self.samples, feed_dict={
                                self.pz_samples: fixed_noise[:5],
                                self.sigma_scale: self.opts['sigma_scale_stochasticity'][n]})
                Samples.append(samples)
            plot_stochasticity(self.opts, Samples, exp_dir, 'test_plots', 'stochasticity.png' % it)


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
        # mnist plot setup
        num_cols = 10
        num_pics = num_cols**2
        num_pics_enc = 5000
        data_ids = np.random.choice(200, 18, replace=True)
        # svhn plot setup
        # num_cols = 18
        # num_pics = 7*num_cols
        # # num_pics = num_cols*num_cols
        # num_pics_enc = 200
        # data_ids = np.random.choice(200, 18, replace=True)
        # data_ids = [21, 7, 24, 18, 28, 12, 75, 82, 32]
        # --- Reconstructions
        full_recons = self.sess.run(self.full_reconstructed,
                                feed_dict={self.points:data.test_data[:200], #num_pics],
                                           self.dropout_rate: 1.,
                                           self.is_training:False})
        reconstructed = full_recons[-1]
        # mnist plot setup
        full_recon = [full_recons[i][data_ids] for i in range(len(full_recons))]
        full_reconstructed = [data.test_data[data_ids],] + full_recon
        # svhn plot setup
        # full_recon = [full_recons[i][data_ids] for i in range(len(full_recons))]
        # full_reconstructed = [data.test_data[data_ids],] + full_recon
        # full_recon = [full_recons[i][data_ids] for i in range(len(full_recons))]
        # full_reconstructed = [data.test_data[data_ids],] + full_recon


        # --- Encode anchors points and interpolate
        encoded = self.sess.run(self.encoded,
                                feed_dict={self.points:data.test_data[:num_pics_enc],
                                           self.dropout_rate: 1.,
                                           self.is_training:False})

        encshape = list(np.shape(encoded[-1])[1:])
        # mnist plot setup
        num_steps = 23
        num_anchors = 12
        anchors_ids = np.random.choice(num_pics, num_anchors,
                                        replace=True)
        data_anchors = data.test_data[anchors_ids]
        # svhn plot setup
        # num_steps = 16
        # num_anchors = 14
        # anchors_ids = np.random.choice(num_pics, num_anchors,
        #                                 replace=True)
        # data_anchors = data.test_data[anchors_ids]

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
            if False:
                enc_mean = np.mean(encoded[-1],axis=0)
                enc_var = np.mean(np.square(encoded[-1]-enc_mean),axis=0)
            else:
                enc_mean = np.zeros(opts['zdim'][-1], dtype='float32')
                enc_var = np.ones(opts['zdim'][-1], dtype='float32')
            mins, maxs = enc_mean - 2.*np.sqrt(enc_var), enc_mean + 2.*np.sqrt(enc_var)
            x = np.linspace(mins[0], maxs[0], num=num_cols, endpoint=True)
            xymin = np.stack([x,mins[1]*np.ones(num_cols)],axis=-1)
            xymax = np.stack([x,maxs[1]*np.ones(num_cols)],axis=-1)
            latent_anchors = np.stack([xymin,xymax],axis=1)
            grid_interpolation = linespace(opts, num_cols,
                                    anchors=latent_anchors)
            dec_latent = self.sess.run(self.decoded[-1],
                                    feed_dict={self.samples: np.reshape(grid_interpolation,[-1,]+list(np.shape(enc_mean))),
                                               self.dropout_rate: 1.,
                                               self.is_training: False})
            inter_latent = np.reshape(dec_latent,[-1,num_cols]+imshape)
        else:
            inter_latent = None

        # --- Samples generation
        prior_noise = sample_pz(opts, self.pz_params, num_pics)
        samples = self.sess.run(self.decoded[-1],
                               feed_dict={self.samples: prior_noise,
                                          self.dropout_rate: 1.,
                                          self.is_training: False})
        # --- Making & saving plots
        save_latent_interpolation(opts, data.test_data[:num_pics_enc],data.test_labels[:num_pics_enc], # data,labels
                        encoded, # encoded
                        reconstructed, full_reconstructed, # recon, full_recon
                        inter_anchors, inter_latent, # anchors and latents interpolation
                        samples, # samples
                        MODEL_PATH) # working directory

    def vlae_experiment(self, data, MODEL_PATH, WEIGHTS_FILE):
        """
        Plot and save different latent interpolation
        """

        opts = self.opts
        num_pics = opts['plot_num_pics']
        # num_pics = 16

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
                if opts['d_archi'][n]=='resnet_v2':
                    features_dim=self.features_dim[n+1]
                else:
                    features_dim=self.features_dim[n]
                # Decoding
                decoded_mean, decoded_Sigma = decoder(opts, input=decoded,
                                                archi=opts['d_archi'][n],
                                                num_layers=opts['d_nlayers'][n],
                                                num_units=opts['d_nfilters'][n],
                                                filter_size=opts['filter_size'][n],
                                                output_dim=output_dim,
                                                features_dim=features_dim,
                                                resample=opts['d_resample'][n],
                                                last_archi=opts['d_last_archi'][n],
                                                scope='decoder/layer_%d' % n,
                                                reuse=True,
                                                is_training=False)
                if opts['decoder'][n] == 'det':
                    decoded = decoded_mean
                elif opts['decoder'][n] == 'gauss':
                    if n==opts['nlatents']-m:
                        if m==0:
                            p_params = tf.concat((decoded_mean,decoded_Sigma),axis=-1)
                            decoded = sample_gaussian(opts, p_params, 'tensorflow')
                        else:
                            shape = decoded_mean.get_shape().as_list()[-1]
                            eps = []
                            for i in range(num_pics):
                                # eps.append(sample_unif([shape,],-(3-n/2.),3-n/2.))
                                eps.append(sample_unif([shape,],-1,1))
                            eps = tf.stack(eps,axis=0)
                            decoded = decoded_mean + eps
                    else:
                        decoded =  decoded_mean #+ tf.multiply(fixed_noise[opts['nlatents']-n],tf.sqrt(1e-10+decoded_Sigma))
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
        # logging.error('Saving images..')
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
                                                self.matching_penalty],
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
                                                self.matching_penalty],
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
