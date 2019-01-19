import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils
from math import ceil, log

import tensorflow as tf

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--zdim", type=int, default=100,
                    help='latent dimension',)

FLAGS = parser.parse_args()

def main():

    # Select dataset to use
    opts = configs.config_cifar10

    # training WAE
    opts['method'] = 'wae'
    utils.create_dir(opts['method'])

    # Verbose
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    # Create root_directory
    root_dir = 'cifar10_' + str(FLAGS.zdim)
    root_path = os.path.join(opts['method'],root_dir)
    utils.create_dir(root_path)
    ### Experiments
    for nlayers in range(3,6):
        ## Create experiment sub directories
        sub_dir = str(nlayers) + '_layers'
        sub_path = os.path.join(root_path,sub_dir)
        utils.create_dir(os.path.join(sub_path))
        ## Experiment set up
        # nlatents
        opts['nlatents'] = nlayers
        # Latents dimension
        opts['zdim'] = [FLAGS.zdim,]
        base_2 = ceil(log(FLAGS.zdim)/log(2))
        zdims = [2**(base_2-i) for i in range(1,nlayers)]
        opts['zdim'] += zdims
        # Checking zdim[i] is square for conv layer
        for i in range(len(opts['zdim'])):
            if opts['zdim'][i]==32 and i<len(opts['zdim'])-1:
                opts['zdim'][i]=36
            if opts['zdim'][i]==8 and i<len(opts['zdim'])-1:
                opts['zdim'][i]=9
        # encoder type
        opts['encoder'] = ['gauss' for i in range(nlayers-1)]
        opts['encoder'].append('det')
        ## lambda values
        lambda_values = [10.**i for i in range(1,-2,-1)]
        for lambda_scalar in lambda_values:
            logging.error('%d layers, zdim %d, lambda %.1f' % (nlayers,FLAGS.zdim,lambda_scalar))
            # lambda Value
            opts['lambda_scalar'] = lambda_scalar
            opts['lambda'] = [1. for i in range(len(opts['zdim'])-1)]
            opts['lambda'].append(opts['lambda_scalar'])
            # Working directory
            work_dir = 'lambda_' + str(lambda_scalar)
            work_path = os.path.join(sub_path,work_dir)
            utils.create_dir(work_path)
            # Checkpoints dir
            opts['work_dir'] = work_path
            utils.create_dir(os.path.join(work_path, 'checkpoints'))
            # Dumping all the configs to the text file
            with utils.o_gfile((work_path, 'params.txt'), 'w') as text:
                text.write('Parameters:\n')
                for key in opts:
                    text.write('%s : %s\n' % (key, opts[key]))

            #Reset tf graph
            tf.reset_default_graph()

            # build WAE
            wae = WAE(opts)

            # Training
            wae.train(data, opts['work_dir'], 'none')

main()
