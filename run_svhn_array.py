import os
import sys
import logging
import argparse
import itertools
import tensorflow as tf
from math import exp

import configs
from wae import WAE
# from vae import VAE
from vae_v2 import VAE
from datahandler import DataHandler
import utils

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--penalty", default='wae',
                    help='penalty type [wae/wae_mmd]')
parser.add_argument("--work_dir")
parser.add_argument("--etype", default='gauss',
                    help='encoder type')
parser.add_argument("--enet_archi", default='resnet',
                    help='encoder networks architecture [mlp/dcgan_v2/resnet]')
parser.add_argument("--dnet_archi", default='resnet',
                    help='decoder networks architecture [mlp/dcgan_v2/resnet]')
parser.add_argument("--weights_file")
parser.add_argument('--gpu_id', default='cpu',
                    help='gpu id for DGX box. Default is cpu')
parser.add_argument('--dataset', default='cropped',
                    help='use original cropped or extra dataset')
parser.add_argument('--exp_id', type=int, default=1,
                    help='experiment id')


FLAGS = parser.parse_args()

if FLAGS.gpu_id!='cpu':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

def main():

    # Select dataset to use
    opts = configs.config_svhn

    # Select training method
    opts['method'] = 'wae'

    # Working directory
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir

    # Mode
    if FLAGS.mode=='fid':
        opts['fid'] = True
    else:
        opts['fid'] = False

    # Experiemnts set up
    opts['epoch_num'] = 305
    opts['print_every'] =  50*9443 #extra: 9443it/epoch, cropped: 1144it/epoch
    opts['lr'] = 0.0003
    opts['batch_size'] = 64
    opts['dropout_rate'] = 1.
    opts['rec_loss_resamples'] = 'encoder'
    opts['rec_loss_nsamples'] = 1
    opts['save_every'] = 10000000*6043
    opts['save_final'] = False #True
    opts['save_train_data'] = True
    opts['use_trained'] = False
    opts['vizu_encSigma'] = True

    # Model set up
    opts['nlatents'] = 6
    opts['zdim'] = [2, 1, 2, 1, 2, 16]
    # opts['zdim'] = [3, 2, 1, 2, 1, 32]
    # zdims = [[2, 1, 2, 1, 2, 16], [1, 3, 2, 1, 2, 16], [1, 2, 4, 2, 1, 8],[2, 1, 2, 1, 2, 16]]
    # opts['zdim'] = zdims[FLAGS.exp_id-1]

    # Penalty
    opts['pen'] = FLAGS.penalty
    opts['mmd_kernel'] = 'IMQ'
    lmbas = []
    base_lmba = [0.5,]
    # lmba = [5., 10., 50.]
    lmba = [5., 10., 20., 40., 50., 60., 100.]
    pen_sigma_coef = [6./6., 5./6., 4./6.]
    lmbas += list(itertools.product(base_lmba,lmba,pen_sigma_coef))
    base_lmba = [1.,]
    # lmba = [5., 10., 50.]
    lmba = [100., 200., 400., 500., 600., 1000.]
    pen_sigma_coef = [5./6., 4./6., 3./6]
    lmbas += list(itertools.product(base_lmba,lmba,pen_sigma_coef))
    # base_lmba = [0.01, 0.05, 0.1, 0.5]
    # lmba1 = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    # lmbas = list(itertools.product(base_lmba1,lmba1))
    opts['lambda'] = [lmbas[FLAGS.exp_id-1][0][0]**(i/1.+1) for i in range(opts['nlatents']-1)]
    opts['lambda'].append(lmbas[FLAGS.exp_id-1][0][1])
    opts['pen_enc_sigma'] = True
    # opts['lambda_pen_enc_sigma'] = [2.5 * exp(-5. * i / 6.) for i in range(opts['nlatents'])]
    opts['lambda_pen_enc_sigma'] = [2.5 * exp(- lmbas[FLAGS.exp_id-1][1]* i) for i in range(opts['nlatents'])]
    opts['lambda_pen_enc_sigma'][-1] *= 1.5
    opts['pen_dec_sigma'] = False
    opts['lambda_pen_dec_sigma'] = [0.0005,]*opts['nlatents']
    opts['obs_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
    opts['latent_cost'] = 'l2sq_gauss' #l2, l2sq, l2sq_norm, l2sq_gauss, l1

    # NN set up
    opts['filter_size'] = [5,3,3,3,3,3,3,3,3,3]
    opts['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
    opts['e_nlatents'] = opts['nlatents']
    opts['encoder'] = [FLAGS.etype,]*opts['nlatents'] # deterministic, gaussian
    opts['e_arch'] = [FLAGS.enet_archi,]*opts['nlatents'] # mlp, dcgan, dcgan_v2, resnet
    opts['e_last_archi'] = ['conv',]*opts['nlatents'] # dense, conv1x1, conv
    # opts['e_resample'] = ['down', None, None, 'down', None, 'down'] # None, down
    opts['e_resample'] = ['down', None, 'down', None, 'down', None]# None, down
    opts['e_nlayers'] = [3,]*opts['nlatents']
    # opts['e_nfilters'] = [64, 64, 64, 128, 128, 128]
    opts['e_nfilters'] = [64, 64, 96, 96, 128, 128]
    opts['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
    opts['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
    opts['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
    opts['d_arch'] =  [FLAGS.dnet_archi,]*opts['nlatents'] # mlp, dcgan, dcgan_mod, resnet
    opts['d_last_archi'] = ['conv',]*opts['nlatents'] # dense, conv1x1, conv
    opts['d_resample'] = ['up', None, 'up', None, 'up', None] #None, up
    # opts['d_resample'] = ['up', None, None, 'up', None, 'up'] #None, up
    opts['d_nlayers'] = [3,]*opts['nlatents']
    # opts['d_nfilters'] = [64, 64, 64, 128, 128, 128]
    opts['d_nfilters'] = [64, 64, 96, 96, 128, 128]
    opts['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
    opts['d_norm'] = 'batchnorm' #batchnorm, layernorm, none

    # Create directories
    if not tf.gfile.IsDirectory(opts['method']):
        utils.create_dir(opts['method'])
    out_dir = os.path.join(opts['method'],opts['dataset'])
    if not tf.gfile.IsDirectory(out_dir):
        utils.create_dir(out_dir)
    work_dir = os.path.join(out_dir,opts['work_dir'])
    opts['work_dir'] = work_dir
    if not tf.gfile.IsDirectory(work_dir):
        utils.create_dir(work_dir)
        utils.create_dir(os.path.join(work_dir, 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(work_dir,'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    if FLAGS.dataset=='extra':
        opts['use_extra']=True
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
