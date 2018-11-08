import sys
import time
import os
import argparse
import logging

from math import sqrt, cos, sin, pi
import numpy as np
import scipy.stats as scistats
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import umap

import configs
from datahandler import DataHandler
import utils

import pdb

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)

def calculate_row_entropy(mean_probs):
    entropies = []
    for i in range(np.shape(mean_probs)[0]):
        entropies.append(scistats.entropy(mean_probs[i]))
    entropies = np.asarray(entropies)
    return entropies

def relabelling_mask(mean_probs, entropies):
    k_vals = []
    max_entropy_state = np.ones(len(entropies))/len(entropies)
    max_entropy = scistats.entropy(max_entropy_state)
    mask = np.arange(10)
    while np.amin(entropies) < max_entropy:
        digit_idx = np.argmin(entropies)
        k_val_sort = np.argsort(mean_probs[digit_idx])
        i = -1
        k_val = k_val_sort[i]
        while k_val in k_vals:
            i -= 1
            k_val = k_val_sort[i]
        k_vals.append(k_val)
        mask[k_val] = digit_idx
        entropies[digit_idx] = max_entropy
    return mask

def relabelling_mask_from_probs(mean_probs):
    probs_copy = mean_probs.copy()
    nmixtures = np.shape(mean_probs)[-1]
    k_vals = []
    min_prob = np.zeros(nmixtures)
    mask = np.arange(10)
    while np.amax(probs_copy) > 0.:
        max_probs = np.amax(probs_copy,axis=-1)
        digit_idx = np.argmax(max_probs)
        k_val_sort = np.argsort(probs_copy[digit_idx])
        i = -1
        k_val = k_val_sort[i]
        while k_val in k_vals:
            i -= 1
            k_val = k_val_sort[i]
        k_vals.append(k_val)
        mask[k_val] = digit_idx
        probs_copy[digit_idx] = min_prob
    return mask


def plots_test(data_train,
                rec_train,
                data_test, label_test,
                encoded, enc_mean, rec_test, prob,
                anchors,
                decod_inteprolation,
                sample_prior,
                sample_gen,
                prior_decod_interpolation,
                work_dir):
    """ Generates and saves the following plots:
        img1    -   train reconstruction
        img2    -   test reconstruction
        img3    -   samples
        img4    -   test interpolation
        img5    -   prior interpolation
        img6    -   discrete latents
        img7    -   UMAP
    """

    if not tf.gfile.IsDirectory(work_dir):
        raise Exception("working directory doesnt exist")
    save_dir = os.path.join(work_dir,'figures')
    utils.create_dir(save_dir)

    greyscale = np.shape(prior_decod_interpolation)[-1] == 1
    zdim = np.shape(encoded)[-1]

    #if opts['input_normalize_sym']:
    if False:
        data_train = data_train / 2. + 0.5
        rec_train = rec_train / 2. + 0.5
        data_test = data_test / 2. + 0.5
        anchors = anchors / 2. + 0.5
        decod_inteprolation = decod_inteprolation / 2. + 0.5
        sample_gen = sample_gen / 2. + 0.5
        prior_decod_interpolation = prior_decod_interpolation / 2. + 0.5

    images = []

    ### Reconstruction plots
    for pair in [(data_train, rec_train),
                 (data_test, rec_test)]:
        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        num_pics = np.shape(sample)[0]
        num_cols = 20
        assert len(sample) == len(recon)
        pics = []
        merged = np.vstack([recon, sample])
        r_ptr = 0
        w_ptr = 0
        for _ in range(int(num_pics / 2)):
            merged[w_ptr] = sample[r_ptr]
            merged[w_ptr + 1] = recon[r_ptr]
            r_ptr += 1
            w_ptr += 2
        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - merged[idx, :, :, :])
            else:
                pics.append(merged[idx, :, :, :])
        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)


    ### Points Interpolation plots
    white_pix = 4
    num_pics = np.shape(decod_inteprolation)[0]
    num_cols = np.shape(decod_inteprolation)[1]
    pics = []
    for idx in range(num_pics):
        if greyscale:
            pic = 1. - decod_inteprolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=2)
            white = np.zeros((white_pix,)+np.shape(pic)[2:])
            pic = np.concatenate((white,pic[0]),axis=0)
            pics.append(pic)
        else:
            pic = decod_inteprolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=1)
            white = np.zeros((white_pix,)+np.shape(pic)[1:])
            pic = np.concatenate(white,pic)
            pics.append(pic)
    image = np.concatenate(pics, axis=0)
    images.append(image)

    ###Prior Interpolation plots
    white_pix = 4
    num_pics = np.shape(prior_decod_interpolation)[0]
    num_cols = np.shape(prior_decod_interpolation)[1]
    pics = []
    for idx in range(num_pics):
        if greyscale:
            pic = 1. - prior_decod_interpolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=2)
            if zdim!=2:
                white = np.zeros((white_pix,)+np.shape(pic)[2:])
                pic = np.concatenate((white,pic[0]),axis=0)
            pics.append(pic)
        else:
            pic = prior_decod_interpolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=1)
            if zdim!=2:
                white = np.zeros((white_pix,)+np.shape(pic)[1:])
                pic = np.concatenate(white,pic)
            pics.append(pic)
    # Figuring out a layout
    image = np.concatenate(pics, axis=0)
    images.append(image)

    img1, img2, img3, img4 = images

    ###Settings for pyplot fig
    dpi = 100
    for img, title, filename in zip([img1, img2, img3, img4],
                         ['Train reconstruction',
                         'Test reconstruction',
                         'Points interpolation',
                         'Priors interpolation'],
                         ['train_recon',
                         'test_recon',
                         'point_inter',
                         'prior_inter']):
        height_pic = img.shape[0]
        width_pic = img.shape[1]
        fig_height = height_pic / 10
        fig_width = width_pic / 10
        fig = plt.figure(figsize=(fig_width, fig_height))
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing axes, ticks, labels
        plt.axis('off')
        # # placing subplot
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
        # Saving
        filename = filename + '.png'
        plt.savefig(utils.o_gfile((save_dir, filename), 'wb'),
                    dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
        plt.close()


    # Set size for following plots
    height_pic= img1.shape[0]
    width_pic = img1.shape[1]
    fig_height = height_pic / float(dpi)
    fig_width = width_pic / float(dpi)


    ###The mean mixtures plots
    mean_probs = []
    num_pics = np.shape(prob)[0]
    for i in range(10):
        prob = [prob[k] for k in range(num_pics) if label_test[k]==i]
        prob = np.mean(np.stack(prob,axis=0),axis=0)
        mean_probs.append(prob)
    mean_probs = np.stack(mean_probs,axis=0)
    # entropy
    entropies = calculate_row_entropy(mean_probs)
    cluster_to_digit = relabelling_mask(mean_probs, entropies)
    digit_to_cluster = np.argsort(cluster_to_digit)
    mean_probs = mean_probs[::-1,digit_to_cluster]
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mean_probs,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    plt.title('Average probs')
    plt.yticks(np.arange(10),np.arange(10)[::-1])
    plt.xticks(np.arange(10))
    # Saving
    filename = 'probs.png'
    fig.savefig(utils.o_gfile((save_dir, filename), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()


    ###Sample plots
    pics = []
    num_cols = 10
    num_pics = np.shape(sample_gen)[0]
    size_pics = np.shape(sample_gen)[1]
    num_to_keep = 20
    for idx in range(num_pics):
        if greyscale:
            pics.append(1. - sample_gen[idx, :, :, :])
        else:
            pics.append(sample_gen[idx, :, :, :])
    # Figuring out a layout
    pics = np.array(pics)
    cluster_pics = np.array(np.split(pics, num_cols))[digit_to_cluster]
    img = np.concatenate(cluster_pics.tolist(), axis=2)
    img = np.concatenate(img, axis=0)
    img = img[:num_to_keep*size_pics]
    fig = plt.figure(figsize=(img.shape[1]/10, img.shape[0]/10))
    if greyscale:
        image = img[:, :, 0]
        # in Greys higher values correspond to darker colors
        plt.imshow(image, cmap='Greys',
                        interpolation='none', vmin=0., vmax=1.)
    else:
        plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
    # Removing axes, ticks, labels
    plt.axis('off')
    # # placing subplot
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    # Saving
    filename = 'gen_sample.png'
    plt.savefig(utils.o_gfile((save_dir, filename), 'wb'),
                dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
    plt.close()


    ###UMAP visualization of the embedings
    num_pics = 200
    if np.shape(encoded)[-1]==2:
        embedding = np.concatenate((encoded,enc_mean,sample_prior),axis=0)
    else:
        embedding = umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation').fit_transform(np.concatenate((encoded[:num_pics],enc_mean[:num_pics],sample_prior),axis=0))
    fig_height = height_pic / float(dpi)
    fig_width = width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(embedding[:num_pics, 0], embedding[:num_pics, 1],
               c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='Vega10'))
    plt.colorbar()
    plt.scatter(embedding[num_pics:(2*num_pics-1), 0], embedding[num_pics:(2*num_pics-1), 1],
               color='aqua', s=3, alpha=0.5, marker='x',label='mean Qz test')
    plt.scatter(embedding[2*num_pics:, 0], embedding[2*num_pics:, 1],
                            color='navy', s=3, alpha=0.5, marker='*',label='Pz')
    xmin = np.amin(embedding[:,0])
    xmax = np.amax(embedding[:,0])
    magnify = 0.1
    width = abs(xmax - xmin)
    xmin = xmin - width * magnify
    xmax = xmax + width * magnify
    ymin = np.amin(embedding[:,1])
    ymax = np.amax(embedding[:,1])
    width = abs(ymin - ymax)
    ymin = ymin - width * magnify
    ymax = ymax + width * magnify
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off',
                    right='off',
                    left='off',
                    labelleft='off')
    plt.legend(loc='upper left')
    plt.title('UMAP latents')
    # Saving
    filename = 'umap.png'
    fig.savefig(utils.o_gfile((save_dir, filename), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()


    ###Saving plots and data
    data_dir = 'data_for_plots'
    save_path = os.path.join(work_dir,data_dir)
    utils.create_dir(save_path)
    filename = 'final_plots.npy'
    np.savez(os.path.join(save_path,filename),
                smples=sample_gen,
                smples_pr=sample_prior,
                rec_tr=rec_train,
                rec_te=rec_test,
                enc=encoded,
                enc_mean=enc_mean,
                points=decod_inteprolation,
                priors=prior_decod_interpolation,
                prob=prob)

def plots_train(losses,
                data_test,
                labels_test,
                recons,
                enc_mean_all, prob,
                work_dir,
                filename):
    """ Generates and saves the following plots:
        img1    -   losses curves
        img2    -   q(k|X)
        img3    -   umap q(z|k,x)
    """

    if not tf.gfile.IsDirectory(work_dir):
        raise Exception("working directory doesnt exist")
    path_dir = os.path.join(work_dir,'figures')
    save_dir = os.path.join(path_dir,'train')
    utils.create_dir(save_dir)

    greyscale = np.shape(data_test)[-1] == 1
    zdim = np.shape(enc_mean_all)[-1]
    dpi = 100

    ### Reconstruction plots
    sample = recons['test_data']
    recon = recons['rec_test']
    num_pics = np.shape(sample)[0]
    num_cols = 10
    assert len(sample) == len(recon)
    pics = []
    merged = np.vstack([recon, sample])
    r_ptr = 0
    w_ptr = 0
    for _ in range(int(num_pics / 2)):
        merged[w_ptr] = sample[r_ptr]
        merged[w_ptr + 1] = recon[r_ptr]
        r_ptr += 1
        w_ptr += 2
    for idx in range(num_pics):
        if greyscale:
            pics.append(1. - merged[idx, :, :, :])
        else:
            pics.append(merged[idx, :, :, :])
    # Figuring out a layout
    pics = np.array(pics)
    img = np.concatenate(np.split(pics, num_cols), axis=2)
    img = np.concatenate(img, axis=0)
    title = 'test reconstruction'
    name = 'test_recon'
    height_pic = img.shape[0]
    width_pic = img.shape[1]
    fig_height = height_pic / dpi
    fig_width = width_pic / dpi
    fig = plt.figure(figsize=(fig_width, fig_height))
    if greyscale:
        image = img[:, :, 0]
        # in Greys higher values correspond to darker colors
        plt.imshow(image, cmap='Greys',
                        interpolation='none', vmin=0., vmax=1.)
    else:
        plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
    # Removing axes, ticks, labels
    plt.axis('off')
    # # placing subplot
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    plt.title(title)
    # Saving
    name = 'recon_' + filename + '.png'
    fig.savefig(utils.o_gfile((save_dir, name), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()

    ### Loss curves
    loss = losses['loss']
    loss_rec = losses['loss_rec']
    loss_match = losses['loss_match']
    kl_cont = losses['kl_cont']
    kl_disc = losses['kl_disc']
    total_num = len(loss_rec)
    x_step = max(int(total_num / 100), 1)
    x = np.arange(1, len(loss_rec) + 1, x_step)
    fig = plt.figure()
    plt.plot(x, np.log(loss[::x_step]), linewidth=3, color='black', label='log(loss)')
    plt.plot(x, np.log(loss_rec[::x_step]), linewidth=2, color='red', label='log(rec loss)')
    if len(kl_cont)>0:
        plt.plot(x, np.log(loss_match[::x_step]), linewidth=2, color='blue', label='log(match loss)')
        plt.plot(x, np.log(kl_cont[::x_step]), linewidth=1, color='blue', linestyle=':', label='log(cont KL)')
        plt.plot(x, np.log(kl_disc[::x_step]), linewidth=1, color='blue', linestyle='--', label='log(disc KL)')
    else:
        plt.plot(x, np.log(opts['lambda']*np.abs(loss_match[::x_step])), linewidth=2, color='blue', label='log(|match loss|)')
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.title('log losses')
    # sacing
    name = 'loss_' + filename + '.png'
    fig.savefig(utils.o_gfile((save_dir, name), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()

    ### Convergence plots
    num_to_plot = 10
    start = 25
    im_to_plot = 1. - data_test[start:start+num_to_plot]
    prob_to_plot = prob[start:start+num_to_plot]
    ###Discrete prob
    stacked_im = np.reshape(im_to_plot,(-1,28,1))
    height_pic = stacked_im.shape[0]
    width_pic = stacked_im.shape[1]
    # Creating plot
    fig = plt.figure(figsize=(fig_height, 2 * fig_width))
    #fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 10])
    # Input
    plt.subplot(gs[0, 0])
    ax = plt.imshow(stacked_im[:,:,0], cmap='Greys',
                    interpolation='none', vmin=0., vmax=1.)
    ax = plt.subplot(gs[0, 0])
    plt.text(0.47, 1., "x",
             ha="center", va="bottom", size=10, transform=ax.transAxes)
    # Removing ticks
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.set_xlim([0, width_pic])
    ax.axes.set_ylim([height_pic, 0])
    ax.axes.set_aspect(1)
    # heatmap
    ax = plt.subplot(gs[0, 1])
    plt.imshow(prob_to_plot,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    plt.text(0.47, 1., "Q(.|x)",
             ha="center", va="bottom", size=10, transform=ax.transAxes)
    plt.xticks(np.arange(10),np.arange(10))
    plt.yticks([])
    # saving
    name = 'probs_' + filename + '.png'
    fig.savefig(utils.o_gfile((save_dir, name), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()

    height_pic= 28*20
    width_pic = 28*20
    dpi = 100

    fig_height = height_pic / float(dpi)
    fig_width = width_pic / float(dpi)

    ###The mean mixtures plots
    mean_probs = []
    num_pics = np.shape(prob)[0]
    for i in range(10):
        probs = [prob[k] for k in range(num_pics) if labels_test[k]==i]
        probs = np.mean(np.stack(probs,axis=0),axis=0)
        mean_probs.append(probs)
    mean_probs = np.stack(mean_probs,axis=0)
    # entropy
    #entropies = calculate_row_entropy(mean_probs)
    #cluster_to_digit = relabelling_mask_from_entropy(mean_probs, entropies)
    cluster_to_digit = relabelling_mask_from_probs(mean_probs)
    digit_to_cluster = np.argsort(cluster_to_digit)
    mean_probs = mean_probs[::-1,digit_to_cluster]
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mean_probs,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    plt.title('Average probs')
    plt.yticks(np.arange(10),np.arange(10)[::-1])
    plt.xticks(np.arange(10))
    # saving
    name = 'probs_' + filename + '.png'
    fig.savefig(utils.o_gfile((save_dir, name), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()

def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg",
                        help='algo [swae/vae]')
    parser.add_argument("--plot", default='test',
                        help='plots to make [test/train]')
    parser.add_argument("--exp", default='mnist',
                        help='dataset [mnist/celebA/dsprites]')
    parser.add_argument("--work_dir")
    FLAGS = parser.parse_args()

    # Config
    opts = configs.config_mnist

    # Loading the dataset
    data = DataHandler(opts)

    # data for plots dir
    work_dir_path = os.path.join(FLAGS.alg,FLAGS.work_dir)
    data_dir = os.path.join(work_dir_path,'data_for_plots')
    if not tf.gfile.IsDirectory(data_dir):
        raise Exception("data directory doesnt exist")

    if FLAGS.plot=="train":
        loss_dir = os.path.join(data_dir,'loss')
        if not tf.gfile.IsDirectory(loss_dir):
            raise Exception("loss directory doesnt exist")
        files = os.listdir(loss_dir)
        probs_dir = os.path.join(data_dir,'probs')
        if not tf.gfile.IsDirectory(probs_dir):
            raise Exception("probs directory doesnt exist")
        means_dir = os.path.join(data_dir,'means')
        if not tf.gfile.IsDirectory(means_dir):
            raise Exception("means directory doesnt exist")
        recons_dir = os.path.join(data_dir,'recon')
        if not tf.gfile.IsDirectory(recons_dir):
            raise Exception("recons directory doesnt exist")

        assert len(files)==len(os.listdir(probs_dir)), \
                'Not as many loss files as probs files'
        assert len(files)==len(os.listdir(means_dir)), \
                'Not as many loss files as means files'
        assert len(files)==len(os.listdir(recons_dir)), \
                'Not as many loss files as recons files'

        logging.error('Plotting train')
        for filename in files:
            plots_train(np.load(os.path.join(loss_dir,filename)),
                        data.test_data,
                        data.test_labels,
                        np.load(os.path.join(recons_dir,filename)),
                        np.load(os.path.join(means_dir,filename)[:-3]+'npy'),
                        np.load(os.path.join(probs_dir,filename)[:-3]+'npy'),
                        work_dir_path,
                        filename[:-4])
    elif FLAGS.plot=="test":
        logging.error('Plotting test')
        plots_test(data_train,
                        rec_train,
                        data_test, label_test,
                        encoded, enc_mean, rec_test, prob,
                        anchors,
                        decod_inteprolation,
                        sample_prior,
                        sample_gen,
                        prior_decod_interpolation,
                        work_dir)

if __name__ == '__main__':
    main()
