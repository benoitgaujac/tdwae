import os
import sys
import logging
import argparse
import configs
from onelayer_wae import onelayer_WAE
from datahandler import DataHandler
import utils

import tensorflow as tf

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--penalty", default='wae',
                    help='penalty type [wae/wae_mmd]')
parser.add_argument("--work_dir")
parser.add_argument("--lmba", type=float, default=100.,
                    help='lambda')
parser.add_argument("--base_lmba", type=float, default=1.,
                    help='base lambda')
parser.add_argument("--etype", default='gauss',
                    help='encoder type')
parser.add_argument("--weights_file")

FLAGS = parser.parse_args()


def main():

    # Select dataset to use
    opts = configs.config_mnist

    # Select training method
    opts['method'] = 'wae'

    # Working directory
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir

    # Experiemnts set up
    opts['epoch_num'] = 2010
    opts['print_every'] =  200*469
    opts['lr'] = 0.003
    opts['batch_size'] = 128
    opts['dropout_rate'] = 1.
    opts['rec_loss_resamples'] = 'encoder'
    opts['rec_loss_nsamples'] = 1
    opts['save_every'] = 2000*500
    opts['save_final'] = True
    opts['save_train_data'] = False
    opts['use_trained'] = False
    opts['vizu_encSigma'] = True

    # Model set up
    opts['nlatents'] = 4
    opts['zdim'] = [512, 128, 32, 8]

    # Penalty
    opts['pen'] = FLAGS.penalty
    opts['mmd_kernel'] = 'IMQ'
    opts['pen_enc_sigma'] = False
    opts['lambda_pen_enc_sigma'] = [10.**i for i in range(-6,-(6+opts['nlatents']),-1)]
    opts['lambda_pen_enc_sigma'].append(0.)
    opts['pen_dec_sigma'] = False
    opts['lambda_pen_dec_sigma'] = [0.0005,]*opts['nlatents']
    opts['obs_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
    opts['latent_cost'] = 'l2sq_gauss' #l2, l2sq, l2sq_norm, l2sq_gauss, l1
    #opts['lambda'] = [FLAGS.base_lmba**(i+1) / opts['zdim'][i] for i in range(opts['nlatents']-1)]
    opts['lambda'] = [FLAGS.base_lmba**(i+1) for i in range(opts['nlatents']-1)]
    opts['lambda'].append(FLAGS.lmba)
    opts['lambda_schedule'] = 'constant'

    # NN set up
    opts['filter_size'] = [5,3,3,3,3,3,3,3,3,3]
    opts['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
    opts['last_archi'] = ['conv',]*opts['nlatents'] # dense, conv1x1, conv
    opts['e_nlatents'] = opts['nlatents']
    opts['encoder'] = [FLAGS.etype,]*opts['nlatents'] # deterministic, gaussian
    opts['e_resample'] = ['down',None,None,None,'down','down'] # None, down
    opts['e_nlayers'] = [2,]*opts['nlatents']
    opts['e_nfilters'] = [512,128,32,8]
    opts['e_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
    opts['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
    opts['decoder'] = ['det',]*opts['nlatents'] # deterministic, gaussian
    opts['d_resample'] = ['up',None,None,None,'up','up'] #None, up
    opts['d_nlayers'] = [2,]*opts['nlatents']
    opts['d_nfilters'] = [512,128,32,8]
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
    logging.basicConfig(filename=os.path.join(work_dir,'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    #Reset tf graph
    tf.reset_default_graph()

    # build WAE/VAE
    wae = onelayer_WAE(opts)

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
