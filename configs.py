import copy
from math import pow, sqrt

# MNIST config from ICLR paper

config_mnist = {}
# Outputs set up
config_mnist['verbose'] = False
config_mnist['save_every_epoch'] = 1000
config_mnist['print_every'] = 98 #2400
config_mnist['work_dir'] = 'results_mnist'
config_mnist['plot_num_pics'] = 100
config_mnist['plot_num_cols'] = 10

# Data set up
config_mnist['dataset'] = 'mnist'
config_mnist['data_dir'] = 'mnist'
config_mnist['input_normalize_sym'] = False
config_mnist['MNIST_data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'
config_mnist['Zalando_data_source_url'] = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

# Experiment set up
config_mnist['train_dataset_size'] = 5000
config_mnist['batch_size'] = 128
config_mnist['epoch_num'] = 100
config_mnist['method'] = 'wae' #vae, wae
config_mnist['use_trained'] = False #train from pre-trained model
config_mnist['e_pretrain'] = False #pretrained the encoder parameters
config_mnist['e_pretrain_sample_size'] = 200

# Opt set up
config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 0.0005
config_mnist['lr_adv'] = 0.0008
config_mnist['batch_norm'] = True
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_decay'] = 0.9

# Objective set up
config_mnist['cost'] = 'l2sq' #l2, l2sq, l1
config_mnist['penalty'] = 'sinkhorn' #sinkhorn, mmd
config_mnist['sqrt_MMD'] = False #use MMD estimator or square MMD estimator
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_mnist['MMDpp'] = True #MMD++ as in https://github.com/tolstikhin/wae/blob/master/improved_wae.py
config_mnist['epsilon'] = 0.5 #Sinkhorn regularization parameters
config_mnist['L'] = 50 #Sinkhorn iteration
config_mnist['lambda'] = [100.,]
config_mnist['lambda_schedule'] = 'constant' #  adaptive, constant

# Model set up
config_mnist['nlatents'] = 1
config_mnist['zdim'] = [16,]

# NN set up
config_mnist['conv_filters_dim'] = 3
config_mnist['init_std'] = 0.0099999
config_mnist['init_bias'] = 0.0

config_mnist['e_arch'] = 'dcgan' # mlp, dcgan, ali, began
config_mnist['e_nlayers'] = 2
config_mnist['e_nfilters'] = 64

config_mnist['d_arch'] = 'dcgan' # mlp, dcgan, dcgan_mod, ali, began
config_mnist['d_nlayers'] = 2
config_mnist['d_nfilters'] = 64
