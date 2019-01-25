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

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--mode", default='test',
                    help='mode to run [train/test/vizu]')
parser.add_argument("--exp", default='mnist',
                    help='dataset [mnist/cifar10/].'\
                    ' celebA/dsprites Not implemented yet')
parser.add_argument("--method",
                    help='algo to train [wae/vae]')
parser.add_argument("--work_dir")
parser.add_argument("--weights_file")
parser.add_argument("--base_lambda", type=int, default=100,
                    help='base lambda',)


FLAGS = parser.parse_args()

def main():

    # Select dataset to use
    if FLAGS.exp == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.exp == 'celebA_small':
        opts = configs.config_celebA_small
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.exp == 'mnist_small':
        opts = configs.config_mnist_small
    elif FLAGS.exp == 'cifar10':
        opts = configs.config_cifar10
    elif FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'grassli':
        opts = configs.config_grassli
    elif FLAGS.exp == 'grassli_small':
        opts = configs.config_grassli_small
    else:
        assert False, 'Unknown experiment configuration'

    # Select training method and create dir
    if FLAGS.method:
        opts['method'] = FLAGS.method

    # Verbose
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    # Create root directories
    utils.create_dir(opts['method'])

    # Experiments
    lambda_values = [FLAGS.base_lambda**i for i in range(1,-3,-1)]
    for lambda_scalar in lambda_values:
        logging.error('Experiment lambda %d' % lambda_scalar)
        # lambda Value
        opts['lambda_scalar'] = lambda_scalar
        opts['lambda'] = [1. for i in range(len(opts['zdim'])-1)]
        opts['lambda'].append(opts['lambda_scalar'])

        # Create working directories
        work_dir = FLAGS.work_dir + '_' + str(lambda_scalar)
        work_dir = os.path.join(opts['method'],opts['work_dir'])
        opts['work_dir'] = work_dir
        utils.create_dir(work_dir)
        utils.create_dir(os.path.join(work_dir, 'checkpoints'))

        # Dumping all the configs to the text file
        with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))

        #Reset tf graph
        tf.reset_default_graph()

        # build WAE
        wae = WAE(opts)

        # Training/testing/vizu
        if FLAGS.mode=="train":
            wae.train(data, FLAGS.weights_file)
        elif FLAGS.mode=="vizu":
            raise ValueError('To implement')
            wae.vizu(data, opts['work_dir'], FLAGS.weights_file)
        else:
            raise ValueError('To implement')
            wae.test(data, opts['work_dir'], FLAGS.weights_file)

main()
