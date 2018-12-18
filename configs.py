import copy
from math import pow, sqrt

### MNIST config
config_mnist = {}
# Outputs set up
config_mnist['verbose'] = False
config_mnist['save_every_epoch'] = 1000
config_mnist['print_every'] = 30000
config_mnist['vizu_sinkhorn'] = False
config_mnist['vizu_embedded'] = True
config_mnist['vizu_emb'] = 'pca' #vizualisation method of the embeddings: pca, umap
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
config_mnist['train_dataset_size'] = -1
config_mnist['batch_size'] = 128
config_mnist['epoch_num'] = 2001
config_mnist['method'] = 'wae' #vae, wae
config_mnist['use_trained'] = False #train from pre-trained model
config_mnist['e_pretrain'] = False #pretrained the encoder parameters
config_mnist['e_pretrain_it'] = 1000
config_mnist['e_pretrain_sample_size'] = 200

# Opt set up
config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 0.0001
config_mnist['lr_adv'] = 0.0008
config_mnist['batch_norm'] = True
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_decay'] = 0.9

# Objective set up
config_mnist['coef_rec'] = .01 # coef recon loss
config_mnist['cost'] = 'l2sq_norm' #l2, l2sq, l2sq_norm, l1
config_mnist['penalty'] = 'sinkhorn' #sinkhorn, mmd
config_mnist['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_mnist['L'] = 20 #Sinkhorn iteration
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ

# Model set up
config_mnist['nlatents'] = 5
config_mnist['zdim'] = [32,16,8,4,2]
config_mnist['pz_scale'] = 1.
config_mnist['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_mnist['lambda_scalar'] = 200.
config_mnist['lambda'] = [config_mnist['zdim'][i]*config_mnist['lambda_scalar']**(i+1)/784 for i in range(len(config_mnist['zdim'])-1)]
config_mnist['lambda'].append(config_mnist['coef_rec']*config_mnist['lambda_scalar']**config_mnist['nlatents']/784)
config_mnist['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_mnist['conv_filters_dim'] = 3
config_mnist['init_std'] = 0.099999
config_mnist['init_bias'] = 0.0

config_mnist['encoder'] = 'deterministic' # deterministic, gaussian
config_mnist['e_arch'] = 'mlp' # mlp, dcgan, ali, began
config_mnist['e_nlayers'] = 2
config_mnist['e_nfilters'] = [512,256,128,64,32]

config_mnist['decoder'] = 'deterministic' # deterministic, gaussian
config_mnist['d_arch'] = 'mlp' # mlp, dcgan, dcgan_mod, ali, began
config_mnist['d_nlayers'] = 2
config_mnist['d_nfilters'] = [512,256,128,64,32]


### CIFAR 10 config
config_cifar10 = {}
# Outputs set up
config_cifar10['verbose'] = False
config_cifar10['save_every_epoch'] = 1000
config_cifar10['print_every'] = 1000
config_cifar10['vizu_sinkhorn'] = False
config_cifar10['vizu_embedded'] = True
config_cifar10['vizu_emb'] = 'pca' #vizualisation method of the embeddings: pca, umap
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
config_cifar10['lr'] = 0.0001
config_cifar10['lr_adv'] = 0.0008
config_cifar10['batch_norm'] = True
config_cifar10['batch_norm_eps'] = 1e-05
config_cifar10['batch_norm_decay'] = 0.9

# Objective set up
config_cifar10['coef_rec'] = .01 # coef recon loss
config_cifar10['cost'] = 'l2sq_norm' #l2, l2sq, l2sq_norm, l1
config_cifar10['penalty'] = 'sinkhorn' #sinkhorn, mmd
config_cifar10['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_cifar10['L'] = 30 #Sinkhorn iteration
config_cifar10['mmd_kernel'] = 'IMQ' # RBF, IMQ

# Model set up
config_cifar10['nlatents'] = 5
config_cifar10['zdim'] = [121,64,36,16,8]
config_cifar10['pz_scale'] = 1.
config_cifar10['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_cifar10['lambda_scalar'] = 200.
config_cifar10['lambda'] = [config_cifar10['zdim'][i]*config_cifar10['lambda_scalar']**(i+1)/1024 for i in range(len(config_cifar10['zdim'])-1)]
config_cifar10['lambda'].append(config_cifar10['coef_rec']*config_cifar10['lambda_scalar']**config_cifar10['nlatents']/1024)
config_cifar10['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_cifar10['conv_filters_dim'] = 3
config_cifar10['init_std'] = 0.099999
config_cifar10['init_bias'] = 0.0

config_cifar10['encoder'] = 'deterministic' # deterministic, gaussian
config_cifar10['e_arch'] = 'dcgan' # mlp, dcgan, ali, began
config_cifar10['e_nlayers'] = 2
config_cifar10['e_nfilters'] = [128,64,64,32,32] #[64,32,32]

config_cifar10['decoder'] = 'deterministic' # deterministic, gaussian
config_cifar10['d_arch'] = 'dcgan' # mlp, dcgan, dcgan_mod, ali, began
config_cifar10['d_nlayers'] = 2
config_cifar10['d_nfilters'] = [128,64,64,32,32] #[64,32,32]
