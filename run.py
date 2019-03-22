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
parser.add_argument("--lmba", type=float, default=100.,
                    help='lambda')
parser.add_argument("--etype", default='gauss',
                    help='encoder type')
parser.add_argument("--weights_file")


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
        assert False, 'Unknown experiment dataset'

    # Select training method
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
    # Model set up
    opts['nlatents'] = 5
    opts['zdim'] = [32,16,8,4,2]
    opts['lambda'] = [1./opts['zdim'][i] for i in range(opts['nlatents']-1)]
    opts['lambda_scalar'] = FLAGS.lmba
    opts['lambda'].append(FLAGS.lmba / opts['zdim'][-1])
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

    # Create directories
    if not tf.gfile.IsDirectory(opts['method']):
        utils.create_dir(opts['method'])
    work_dir = os.path.join(opts['method'],opts['work_dir'])
    opts['work_dir'] = work_dir
    if not tf.gfile.IsDirectory(work_dir):
        utils.create_dir(work_dir)
        utils.create_dir(os.path.join(work_dir, 'checkpoints'))

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    #Reset tf graph
    tf.reset_default_graph()

    # build WAE
    wae = WAE(opts)

    # Training/testing/vizu
    if FLAGS.mode=="train":
        # Dumping all the configs to the text file
        with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        wae.train(data, FLAGS.weights_file)
    elif FLAGS.mode=="vizu":
        wae.latent_interpolation(data, opts['work_dir'], FLAGS.weights_file)
    else:
        raise ValueError('To implement')
        wae.test(data, opts['work_dir'], FLAGS.weights_file)

main()
