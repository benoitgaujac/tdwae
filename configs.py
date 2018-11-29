import copy
from math import pow, sqrt

### MNIST config
config_mnist = {}
# Outputs set up
config_mnist['verbose'] = False
config_mnist['save_every_epoch'] = 1000
config_mnist['print_every'] = 1200
config_mnist['vizu_sinkhorn'] = False
config_mnist['vizu_embedded'] = True
config_mnist['work_dir'] = 'results_mnist'
config_mnist['plot_num_pics'] = 100
config_mnist['plot_num_cols'] = 10

# Data set up
config_mnist['dataset'] = 'zalando'
config_mnist['data_dir'] = 'zalando'
config_mnist['input_normalize_sym'] = False
config_mnist['MNIST_data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'
config_mnist['Zalando_data_source_url'] = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

# Experiment set up
config_mnist['train_dataset_size'] = -1
config_mnist['batch_size'] = 128
config_mnist['epoch_num'] = 101
config_mnist['method'] = 'wae' #vae, wae
config_mnist['use_trained'] = False #train from pre-trained model
config_mnist['e_pretrain'] = False #pretrained the encoder parameters
config_mnist['e_pretrain_it'] = 1000
config_mnist['e_pretrain_sample_size'] = 200

# Opt set up
config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 0.001
config_mnist['lr_adv'] = 0.0008
config_mnist['batch_norm'] = True
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_decay'] = 0.9

# Objective set up
config_mnist['coef_rec'] = 1. # coef recon loss
config_mnist['cost'] = 'l2sq_norm' #l2, l2sq, l2sq_norm, l1
config_mnist['penalty'] = 'sinkhorn' #sinkhorn, mmd
config_mnist['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_mnist['L'] = 20 #Sinkhorn iteration
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_mnist['lambda'] = [4.0, 21.0, 107.0, 1.0]
config_mnist['lambda_schedule'] = 'constant' # adaptive, constant

# Model set up
config_mnist['nlatents'] = 4
config_mnist['zdim'] = [16,8,4,2]
config_mnist['pz_scale'] = 1.
config_mnist['prior'] = 'dirichlet' # dirichlet or gaussian


# NN set up
config_mnist['conv_filters_dim'] = 3
config_mnist['init_std'] = 0.0099999
config_mnist['init_bias'] = 0.0

config_mnist['encoder'] = 'gaussian' # deterministic, gaussian
config_mnist['e_arch'] = 'mlp' # mlp, dcgan, ali, began
config_mnist['e_nlayers'] = 2
config_mnist['e_nfilters'] = [128,64,32,16]

config_mnist['d_arch'] = 'mlp' # mlp, dcgan, dcgan_mod, ali, began
config_mnist['d_nlayers'] = 2
config_mnist['d_nfilters'] = [128,64,32,16]

### CIFAR 10 config
config_cifar10 = {}
# Outputs set up
config_cifar10['verbose'] = False
config_cifar10['save_every_epoch'] = 1000
config_cifar10['print_every'] = 1200
config_cifar10['vizu_sinkhorn'] = False
config_cifar10['vizu_embedded'] = True
config_cifar10['work_dir'] = 'results_cifar'
config_cifar10['plot_num_pics'] = 100
config_cifar10['plot_num_cols'] = 10

# Data set up
config_cifar10['dataset'] = 'cifar10'
config_cifar10['data_dir'] = 'cifar10'
config_cifar10['input_normalize_sym'] = False
config_cifar10['cifar10_data_source_url'] = 'https://www.cs.toronto.edu/~kriz/'

# Experiment set up
config_cifar10['train_dataset_size'] = -1
config_cifar10['batch_size'] = 128
config_cifar10['epoch_num'] = 101
config_cifar10['method'] = 'wae' #vae, wae
config_cifar10['use_trained'] = False #train from pre-trained model
config_cifar10['e_pretrain'] = False #pretrained the encoder parameters
config_cifar10['e_pretrain_sample_size'] = 200
config_cifar10['e_pretrain_it'] = 1000

# Opt set up
config_cifar10['optimizer'] = 'adam' # adam, sgd
config_cifar10['adam_beta1'] = 0.5
config_cifar10['lr'] = 0.001
config_cifar10['lr_adv'] = 0.0008
config_cifar10['batch_norm'] = True
config_cifar10['batch_norm_eps'] = 1e-05
config_cifar10['batch_norm_decay'] = 0.9

# Objective set up
config_cifar10['coef_rec'] = 1. # coef recon loss
config_cifar10['cost'] = 'l2sq_norm' #l2, l2sq, l2sq_norm, l1
config_cifar10['penalty'] = 'sinkhorn' #sinkhorn, mmd
config_cifar10['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_cifar10['L'] = 20 #Sinkhorn iteration
config_cifar10['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_cifar10['lambda'] = [4.0, 21.0, 107.0, 1.0]
config_cifar10['lambda_schedule'] = 'constant' # adaptive, constant

# Model set up
config_cifar10['nlatents'] = 4
config_cifar10['zdim'] = [32,16,8,4]
config_cifar10['pz_scale'] = 1.
config_cifar10['prior'] = 'dirichlet' # dirichlet or gaussian


# NN set up
config_cifar10['conv_filters_dim'] = 3
config_cifar10['init_std'] = 0.0099999
config_cifar10['init_bias'] = 0.0

config_cifar10['encoder'] = 'gaussian' # deterministic, gaussian
config_cifar10['e_arch'] = 'mlp' # mlp, dcgan, ali, began
config_cifar10['e_nlayers'] = 2
config_cifar10['e_nfilters'] = [256,128,64,32]

config_cifar10['d_arch'] = 'mlp' # mlp, dcgan, dcgan_mod, ali, began
config_cifar10['d_nlayers'] = 2
config_cifar10['d_nfilters'] = [256,128,64,32]
