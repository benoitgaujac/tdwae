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
import itertools

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
parser.add_argument("--params", type=int, default=1,
                    help='params setup')
parser.add_argument("--etype", default='gauss',
                    help='encoder type')
parser.add_argument("--enet_archi", default='resnet',
                    help='encoder networks architecture [mlp/dcgan_v2/resnet]')
parser.add_argument("--dnet_archi", default='resnet',
                    help='decoder networks architecture [mlp/dcgan_v2/resnet]')
parser.add_argument("--weights_file")
parser.add_argument('--gpu_id', default='cpu',
                    help='gpu id for DGX box. Default is cpu')

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
    opts['epoch_num'] = 4011
    opts['print_every'] = 200*469
    opts['lr'] = 0.0001
    opts['dropout_rate'] = 1.
    opts['batch_size'] = 128
    opts['rec_loss_resamples'] = 'encoder'
    opts['rec_loss_nsamples'] = 1
    opts['save_every_epoch'] = 6005
    opts['save_final'] = True
    opts['save_train_data'] = True
    opts['use_trained'] = False
    opts['vizu_encSigma'] = True

    # Model set up
    opts['nlatents'] = 4
    opts['zdim'] = [32,16,8,2]

    # Penalty
    opts['pen'] = FLAGS.penalty
    opts['mmd_kernel'] = 'IMQ'
    opts['pen_enc_sigma'] = False
    opts['lambda_pen_enc_sigma'] = [10.**i for i in range(-6,-(6+opts['nlatents']),-1)]
    opts['lambda_pen_enc_sigma'].append(0.)
    opts['pen_dec_sigma'] = False
    opts['lambda_pen_dec_sigma'] = 0.0005
    opts['obs_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
    opts['latent_cost'] = 'l2sq_gauss' #l2, l2sq, l2sq_norm, l2sq_gauss, l1
    # opts['lambda'] = [FLAGS.base_lmba**(i+1) / opts['zdim'][i] for i in range(opts['nlatents']-1)]
    base_lmbda = [10**-i for i in range(-1,5)]
    lmbda = [10**-i for i in range(1,7)]
    lmbdas = [[base_lmbda[i],lmbda[j]] for i in range(len(base_lmbda)) for j in range(len(lmbda))]
    opts['lambda'] = [lmbdas[FLAGS.params-1][0]**(i+1) for i in range(opts['nlatents']-1)]
    opts['lambda'].append(lmbdas[FLAGS.params-1][1])
    opts['lambda_schedule'] = 'constant'
    opts['lambda_schedule'] = 'constant'

    # NN set up
    opts['filter_size'] = [3,3,3,3,3,3,3,3,3,3]
    opts['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
    opts['e_nlatents'] = opts['nlatents'] #opts['nlatents']
    opts['encoder'] =  ['det','det','det','gauss'] # deterministic, gaussian
    opts['e_arch'] = [FLAGS.enet_archi,]*opts['nlatents'] # mlp, dcgan, dcgan_v2, resnet
    opts['e_last_archi'] = ['conv',]*opts['nlatents'] # dense, conv1x1, conv
    opts['e_resample'] = ['down', None,'down', None, 'down'] #None, down
    opts['e_nlayers'] = [2,]*opts['nlatents']
    opts['e_nfilters'] = [32,16,8,4]
    opts['e_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
    opts['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
    opts['decoder'] = ['det',]*opts['nlatents'] # deterministic, gaussian
    opts['d_arch'] =  [FLAGS.dnet_archi,]*opts['nlatents'] # mlp, dcgan, dcgan_mod, resnet
    opts['d_last_archi'] = ['dense',]*opts['nlatents'] # dense, conv1x1, conv
    opts['d_resample'] = ['up', None,'up', None, 'up'] #None, up
    opts['d_nlayers'] = [2,]*opts['nlatents']
    opts['d_nfilters'] = [32,16,8,4]
    opts['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
    opts['d_norm'] = 'batchnorm' #batchnorm, layernorm, none

    # Create directories
    if not tf.gfile.IsDirectory(opts['method']):
        utils.create_dir(opts['method'])
    work_dir = os.path.join(opts['method'],opts['work_dir'])
    opts['work_dir'] = work_dir
    if not tf.gfile.IsDirectory(work_dir):
        utils.create_dir(work_dir)
        utils.create_dir(os.path.join(work_dir, 'checkpoints'))

    # Verbose
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
        opts['sample_recons'] = False
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
