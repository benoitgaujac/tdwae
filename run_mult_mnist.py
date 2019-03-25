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
parser.add_argument("--work_dir")
parser.add_argument("--etype", default='gauss',
                    help='encoder type')
parser.add_argument("--weights_file")


FLAGS = parser.parse_args()


def main():
    opts = configs.config_mnist


    # Select training method and create dir
    opts['method'] = 'wae'
    # Working directory
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir

    # Experiemnts set up
    opts['epoch_num'] = 4009
    opts['print_every'] = 187500
    opts['lr'] = 0.0005
    opts['save_every_epoch'] = 2005 #4011
    opts['save_final'] = True
    opts['save_train_data'] = True
    opts['use_trained'] = False
    # Model set up
    opts['nlatents'] = 5
    opts['zdim'] = [32,16,8,4,2]
    opts['lambda'] = [1./opts['zdim'][i] for i in range(opts['nlatents']-1)]
    opts['lambda_schedule'] = 'constant'
    # NN set up
    opts['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
    opts['e_nlatents'] = 5
    opts['encoder'] = [FLAGS.etype,]*opts['nlatents'] #['gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
    opts['e_arch'] = ['mlp','mlp','mlp','mlp','mlp','mlp','mlp'] # mlp, dcgan
    opts['e_nlayers'] = [2,2,2,2,2,2,2]
    opts['e_nfilters'] = [512,256,128,64,32,16]
    opts['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
    opts['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','det'] # deterministic, gaussian
    opts['d_arch'] = ['mlp','mlp','mlp','mlp','mlp','mlp','mlp'] # mlp, dcgan, dcgan_mod
    opts['d_nlayers'] = [2,2,2,2,2,2,2]
    opts['d_nfilters'] = [512,256,128,64,32,16]
    opts['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


    # Verbose
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'


    # Experiments
    lambda_values = [0.00001,0.00005,0.0001,0.0005,0.001]
    for lambda_scalar in lambda_values:
        logging.error('Experiment lambda %d' % lambda_scalar)
        # lambda Value
        opts['lambda'].append(lambda_scalar / opts['zdim'][-1])

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
