import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils

import tensorflow as tf

import pdb


def main():

    # Verbose
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Select dataset to use
    opts = configs.config_mnist

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    # weights_file for vae use WAE pretrained
    weights_file = 'swae/mnist_10mix_v7/checkpoints/trained-wae-18000'

    # VAE loop
    opts['method'] = 'vae'
    vae_exps = [1. ,5., 10., 15., 20., 100.]
    #vae_exps = [1. ,10.]
    for beta in vae_exps:
        # Working directory
        opts['work_dir'] = 'mnist_10mix_v7_beta' + str(beta)
        # Create directories
        utils.create_dir(opts['method'])
        work_dir = os.path.join(opts['method'],opts['work_dir'])
        utils.create_dir(work_dir)
        utils.create_dir(os.path.join(work_dir, 'checkpoints'))
        # setting lambda
        opts['lambda'] = beta
        # No pretrained model
        opts['use_trained'] = False
        # Dumping all the configs to the text file
        with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        #Reset tf graph
        tf.reset_default_graph()
        # build WAE
        wae = WAE(opts)
        # Training
        wae.train(data, opts['work_dir'], weights_file)

    # # VAE init with WAE
    # opts['method'] = 'vae'
    # # Working directory
    # opts['work_dir'] = 'mnist_10mix_v7_wae_pretrained/'
    # # Create directories
    # utils.create_dir(opts['method'])
    # work_dir = os.path.join(opts['method'],opts['work_dir'])
    # utils.create_dir(work_dir)
    # utils.create_dir(os.path.join(work_dir, 'checkpoints'))
    # # setting lambda
    # opts['lambda'] = 10.
    # # Use pretrained model
    # opts['use_trained'] = True
    # # Dumping all the configs to the text file
    # with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
    #     text.write('Parameters:\n')
    #     for key in opts:
    #         text.write('%s : %s\n' % (key, opts[key]))
    # #Reset tf graph
    # tf.reset_default_graph()
    # # build WAE
    # wae = WAE(opts)
    # # Training
    # wae.train(data, opts['work_dir'], weights_file)

    # # WAE 1 mixtures
    # opts['method'] = 'swae'
    # # Working directory
    # opts['work_dir'] = 'mnist_1mix_v7'
    # # Create directories
    # utils.create_dir(opts['method'])
    # work_dir = os.path.join(opts['method'],opts['work_dir'])
    # utils.create_dir(work_dir)
    # utils.create_dir(os.path.join(work_dir, 'checkpoints'))
    # # setting lambda
    # opts['lambda'] = 450.
    # # No pretrained model
    # opts['use_trained'] = False
    # # 1 mixture
    # opts['nmixtures'] = 1
    # # Dumping all the configs to the text file
    # with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
    #     text.write('Parameters:\n')
    #     for key in opts:
    #         text.write('%s : %s\n' % (key, opts[key]))
    # #Reset tf graph
    # tf.reset_default_graph()
    # # build WAE
    # wae = WAE(opts)
    # # Training
    # wae.train(data, opts['work_dir'], weights_file)

main()
