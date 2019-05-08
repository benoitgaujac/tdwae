import os
import sys
import logging
import argparse
import configs
from wae import WAE
# from vae import VAE
from vae_v2 import VAE
from datahandler import DataHandler
import utils

import tensorflow as tf

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--exp", default='mnist',
                    help='dataset [mnist/cifar10/].'\
                    ' celebA/dsprites Not implemented yet')
parser.add_argument("--method", default='wae')
parser.add_argument("--penalty", default='wae',
                    help='penalty type [wae/wae_mmd]')
parser.add_argument("--work_dir")
parser.add_argument("--lmba", type=float, default=0.0001,
                    help='lambda')
parser.add_argument("--base_lmba", type=int, default=0,
                    help='base lambda')
parser.add_argument("--lambda_pen_enc_sigma", type=float, default=.0001,
                    help='Encoder Sgima penalization weight')
parser.add_argument("--etype", default='gauss',
                    help='encoder type')
parser.add_argument("--nfilters", type=int, default=1024,
                    help='number of units in mlp network')
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
    if FLAGS.method:
        opts['method'] = FLAGS.method

    # Working directory
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir

    # Mode
    if FLAGS.mode=='fid':
        opts['fid'] = True
    else:
        opts['fid'] = False

    # Experiemnts set up
    opts['epoch_num'] = 5011
    opts['print_every'] = 187500
    opts['lr'] = 0.001
    opts['batch_size'] = 128
    opts['rec_loss_resamples'] = 'encoder'
    opts['rec_loss_nsamples'] = 5
    opts['save_every_epoch'] = 2005
    opts['save_final'] = True
    opts['save_train_data'] = True
    opts['use_trained'] = False
    opts['vizu_encSigma'] = True

    # Model set up
    opts['nlatents'] = 5
    opts['zdim'] = [32,16,8,4,2]

    # Penalty
    opts['pen'] = FLAGS.penalty
    opts['pen_enc_sigma'] = False
    opts['lambda_pen_enc_sigma'] = FLAGS.lambda_pen_enc_sigma
    opts['pen_dec_sigma'] = False
    opts['lambda_pen_dec_sigma'] = 0.0005
    opts['obs_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
    opts['latent_cost'] = 'l2sq_gauss' #l2, l2sq, l2sq_norm, l2sq_gauss, l1
    lambda_values = [0.1,0.5,1.,2.,5.]
    opts['lambda'] = [lambda_values[FLAGS.base_lmba-1]**(i+1) / opts['zdim'][i] for i in range(opts['nlatents']-1)]
    # opts['lambda'] = [FLAGS.base_lmba**(i+1) for i in range(opts['nlatents']-1)]
    opts['lambda'].append(FLAGS.lmba)
    opts['lambda_schedule'] = 'constant'

    # NN set up
    opts['filter_size'] = [3,3,3,3,3,3,3,3,3,3]
    opts['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
    opts['e_nlatents'] = opts['nlatents'] #opts['nlatents']
    opts['encoder'] = [FLAGS.etype,]*opts['nlatents'] #['gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
    opts['e_arch'] = ['mlp','mlp','mlp','mlp','mlp'] # mlp, dcgan, dcgan_v2, resnet
    opts['e_resample'] = ['down', None,'down', None, 'down'] #None, down
    opts['e_nlayers'] = [2,2,2,2,2,2,2,2]
    opts['e_nfilters'] = [FLAGS.nfilters/2**i for i in range(opts['nlatents'])]
    opts['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
    opts['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
    opts['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
    opts['d_arch'] =  ['mlp','mlp','mlp','mlp','mlp'] # mlp, dcgan, dcgan_mod, resnet
    opts['d_resample'] = ['up', None,'up', None, 'up'] #None, up
    opts['d_nlayers'] = [2,2,2,2,2,2,2,2]
    opts['d_nfilters'] = [FLAGS.nfilters/2**i for i in range(opts['nlatents'])]
    opts['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
    opts['d_norm'] = 'layernorm' #batchnorm, layernorm, none

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

    # build WAE/VAE
    if opts['method']=='wae':
        wae = WAE(opts)
    elif opts['method']=='vae':
        wae = VAE(opts)
    else:
        assert False, 'Unknown methdo %s' % opts['method']

    # Training/testing/vizu
    if FLAGS.mode=="train":
        # Dumping all the configs to the text file
        with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        wae.train(data, FLAGS.weights_file)
    elif FLAGS.mode=="vizu":
        opts['rec_loss_nsamples'] = 1
        wae.latent_interpolation(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="fid":
        wae.fid_score(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="test":
        wae.test_losses(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="vlae_exp":
        wae.vlae_experiment(data, opts['work_dir'], FLAGS.weights_file)
    else:
        assert False, 'Unknown mode %s' % FLAGS.mode

main()
