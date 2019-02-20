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
parser.add_argument("--work_dir")
parser.add_argument("--base_lambda", type=int, default=100,
                    help='base lambda')
parser.add_argument("--weights_file")

FLAGS = parser.parse_args()


# Experiment set up
configs.config_mnist['dataset'] = 'mnist'
configs.config_mnist['data_dir'] = 'mnist'
# Model set up
configs.config_mnist['nlatents'] = 5
configs.config_mnist['zdim'] = [32,16,8,4,2]
configs.config_mnist['prior'] = 'implicit' # dirichlet, gaussian or implicit
# NN set up
configs.config_mnist['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
configs.config_mnist['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
configs.config_mnist['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
configs.config_mnist['e_arch'] = ['mlp','mlp','mlp','mlp','mlp','mlp','mlp'] # mlp, dcgan
configs.config_mnist['e_nlayers'] = [2,2,2,2,2,2,2]
configs.config_mnist['e_nfilters'] = [512,256,128,64,32,16]
configs.config_mnist['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
configs.config_mnist['decoder'] = ['det','det','det','det','det','det','det'] # deterministic, gaussian
configs.config_mnist['d_arch'] = ['mlp','mlp','mlp','mlp','mlp','mlp','mlp'] # mlp, dcgan, dcgan_mod
configs.config_mnist['d_nlayers'] = [2,2,2,2,2,2,2]
configs.config_mnist['d_nfilters'] = [512,256,128,64,32,16]
configs.config_mnist['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


def main():


    opts = configs.config_mnist


    # Select training method and create dir
    opts['method'] = 'wae'
    if not tf.gfile.Exists(opts['method']):
        utils.create_dir(opts['method'])

    # Verbose
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    # Experiments
    lambda_values = [FLAGS.base_lambda**i for i in range(2)]
    for lambda_scalar in lambda_values:
        logging.error('Experiment lambda %d' % lambda_scalar)
        # lambda Value
        opts['lambda_scalar'] = lambda_scalar
        # opts['lambda'] = [opts['lambda_scalar']/0.1**i for i in range(opts['nlatents']-1,1,-1)]
        # opts['lambda'] = [1. for i in range(opts['nlatents']-1)]
        # opts['lambda'].append(opts['lambda_scalar'])
        # opts['lambda'] = [lambda_scalar*opts['zdim'][i]/784. for i in range(opts['nlatents'])]
        opts['lambda'] = [lambda_scalar for i in range(opts['nlatents'])]

        # Create working directories
        work_dir = FLAGS.work_dir + '_' + str(lambda_scalar)
        work_dir = os.path.join(opts['method'],work_dir)
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

        # Training
        wae.train(data, FLAGS.weights_file)

main()
