import os
import sys

import logging
import argparse

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--penalty", default='wae',
                    help='penalty type [wae/wae_mmd]')
parser.add_argument("--work_dir")
parser.add_argument("--zdim", type=int, default=0,
                    help='latent dimensions [0/1]')
parser.add_argument("--pen_sigma", type=int, default=0,
                    help='penalty for encoded Sigma [0/1]')
parser.add_argument("--etype", default='gauss',
                    help='encoder type')
parser.add_argument("--enet_archi", default='resnet_v2',
                    help='encoder networks architecture [mlp/dcgan_v2/resnet/resnet_v2]')
parser.add_argument("--dnet_archi", default='resnet_v2',
                    help='decoder networks architecture [mlp/dcgan_v2/resnet/resnet_v2]')
parser.add_argument("--weights_file")
parser.add_argument('--gpu_id', default='cpu',
                    help='gpu id for DGX box. Default is cpu')

FLAGS = parser.parse_args()

if FLAGS.gpu_id!='cpu':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

import configs
from wae import WAE
# from vae import VAE
from vae_v2 import VAE
from datahandler import DataHandler
import utils

import tensorflow as tf


def main():

    # Select dataset to use
    opts = configs.config_cifar10

    # Working directory
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir

    # Mode
    if FLAGS.mode=='fid':
        opts['fid'] = True
    else:
        opts['fid'] = False

    # Experiemnts set up
    opts['epoch_num'] = 5
    opts['print_every'] = 2*500 #every 100 epochs
    opts['lr'] = 0.0002
    opts['batch_size'] = 100
    opts['dropout_rate'] = 1.
    opts['rec_loss_resamples'] = 'encoder'
    opts['rec_loss_nsamples'] = 1
    opts['save_every'] = 3000*500
    opts['save_final'] = True
    opts['save_train_data'] = True
    opts['use_trained'] = False
    opts['vizu_encSigma'] = True

    # Model set up
    opts['nlatents'] = 8
    zdim = [[22, 20, 18, 16, 14, 12, 10, 128], [36, 32, 28, 24, 20, 16, 12, 128]]
    opts['zdim'] = zdim[FLAGS.zdim]

    # Penalty
    opts['pen'] = FLAGS.penalty
    opts['pen_enc_sigma'] = True
    pen_sigma = [10**-5,5*10**-6]
    opts['lambda_pen_enc_sigma'] = [pen_sigma[FLAGS.pen_sigma],]*(opts['nlatents']-1)
    opts['lambda_pen_enc_sigma'].append(pen_sigma[FLAGS.pen_sigma]/10.)
    opts['pen_dec_sigma'] = False
    opts['lambda_pen_dec_sigma'] = [0.0005,]*opts['nlatents']
    opts['obs_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
    opts['latent_cost'] = 'l2sq_gauss' #l2, l2sq, l2sq_norm, l2sq_gauss, l1
    opts['lambda_schedule'] = 'constant'

    # NN set up
    opts['filter_size'] = [5,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    opts['last_archi'] = ['conv1x1',]*opts['nlatents'] # dense, conv1x1, conv
    opts['e_nlatents'] = opts['nlatents'] #opts['nlatents']
    opts['encoder'] = [FLAGS.etype,]*opts['nlatents'] # deterministic, gaussian
    opts['e_arch'] = [FLAGS.enet_archi,]*opts['nlatents'] # mlp, dcgan, dcgan_v2, resnet
    opts['e_resample'] = ['down',None,None,None,'down',None,None,'down'] #None, down
    opts['e_nlayers'] = [3,]*opts['nlatents']
    opts['e_nfilters'] = [128,]*opts['nlatents']
    opts['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
    opts['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
    opts['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
    opts['d_arch'] =  [FLAGS.dnet_archi,]*opts['nlatents'] # mlp, dcgan, dcgan_mod, resnet
    opts['dconv_1x1'] = False
    opts['d_resample'] = ['up',None,None,None,'up',None,None,'up'] #None, up
    opts['d_nlayers'] = [3,]*opts['nlatents']
    opts['d_nfilters'] = [128,]*opts['nlatents']
    opts['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
    opts['d_norm'] = 'batchnorm' #batchnorm, layernorm, none

    # Experiments set up
    base_lmba = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    lmba = [10**i for i in range(-1,-6,-1)]
    # Experiment loop
    for i in range(len(base_lmba)):
        for j in range(len(lmba)):
            opts['lambda'] = [base_lmba[i]**(k/opts['nlatents']+1) for k in range(opts['nlatents']-1)]
            opts['lambda'].append(lmba[j])

            # Create directories
            if not tf.gfile.IsDirectory(opts['method']):
                utils.create_dir(opts['method'])
            work_dir = os.path.join(opts['method'],opts['work_dir'])
            work_dir += '_lb' + str(i) +'_l0' + str(j)
            opts['work_dir'] = work_dir
            if not tf.gfile.IsDirectory(work_dir):
                utils.create_dir(work_dir)
                utils.create_dir(os.path.join(work_dir, 'checkpoints'))

            # Verbose
            logging.basicConfig(filename=os.path.join(work_dir,'outputs.log'),
                level=logging.INFO, format='%(asctime)s - %(message)s')

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
