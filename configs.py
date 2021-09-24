import copy
from math import pow, sqrt, exp

### Default common config
config = {}
# Outputs set up
config['verbose'] = False
config['save_every'] = 10000
config['save_final'] = True
config['save_train_data'] = True
config['print_every'] = 100
config['vizu_splitloss'] = True
config['vizu_fullrec'] = True
config['vizu_embedded'] = True
config['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config['vizu_latent'] = True
config['vizu_encSigma'] = False
config['fid'] = False
config['out_dir'] = 'results'
config['plot_num_pics'] = 100
config['plot_num_cols'] = 10

# Experiment set up
config['train_dataset_size'] = -1
config['batch_size'] = 100
config['it_num'] = 50000
config['model'] = 'stackedwae' #vae, wae, stackedwae, lvae
config['use_trained'] = False #train from pre-trained model
config['pretrain'] = False #pretrained the encoder parameters
config['pretrain_it'] = 1000
config['pretrain_sample_size'] = 200

# Opt set up
config['optimizer'] = 'adam' # adam, sgd
config['adam_beta1'] = 0.9
config['adam_beta1'] = 0.9
config['adam_beta2'] = 0.999
config['lr'] = 0.0001
config['lr_adv'] = 1e-08
config['enorm'] = 'batchnorm' #batchnorm, layernorm, none
config['dnorm'] = 'batchnorm' #batchnorm, layernorm, none
config['batch_norm_eps'] = 1e-05
config['batch_norm_momentum'] = 0.99

# Objective set up
config['obs_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config['latent_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l2sq_gauss, l1
config['penalty'] = 'mmd' #sinkhorn, mmd
config['mmd_kernel'] = 'IMQ' # RBF, IMQ

# Model set up
config['nlatents'] = 5
config['zdim'] = [32,16,8,4,2] #[32,8]
config['pz_scale'] = 1.
config['sigma_scale'] = 4.
config['prior'] = 'gaussian' # dirichlet, gaussian
config['encoder'] = ['gauss',]*config['nlatents'] # det, gaussian
config['decoder'] = ['det',]+['gauss',]*(config['nlatents']-1) # det, gaussian
config['resamples'] = False
config['nresamples'] = 1

# lambda set up
config['lambda_scalar'] = 2.
config['lambda'] = [config['lambda_scalar']**i/config['zdim'][0] for i in range(config['nlatents'])]
config['lambda'].append(0.0001*config['lambda_scalar']**5/config['zdim'][0])
config['lambda_schedule'] = 'constant' # adaptive, constant

# Sigma penalties
config['pen_sigma'] = False # True, False
config['lambda_sigma'] = [exp(1-i) for i in range(config['nlatents'])]

# NN set up
config['init_std'] = 0.099999
config['init_bias'] = 0.0
config['mlpinit'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['convinit'] = 'he' #he, glorot, normilized_glorot, truncated_norm

### MNIST config
config_mnist = config.copy()

# Data set up
config_mnist['dataset'] = 'mnist'
config_mnist['data_dir'] = 'mnist'
config_mnist['input_normalize_sym'] = False
config_mnist['MNIST_data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'
config_mnist['dataset_size'] = 70000
config_mnist['crop_style'] = 'closecrop' # closecrop, resizecrop

# Model set up
config_mnist['nlatents'] = 5
config_mnist['zdim'] = [32,16,8,4,2]
config_mnist['resample'] = True
config_mnist['nresamples'] = 4

# lambda set up
config_mnist['lambda'] = [0.001**n for n in range(1,config_mnist['nlatents']+1)]
config_mnist['lambda_schedule'] = 'constant' # adaptive, constant

# Cov penalties
config_mnist['pen_sigma'] = True # True, False
config_mnist['lambda_sigma'] = [0.01,]*config_mnist['nlatents']

# NN set up
config_mnist['archi'] = ['mlp',]*config['nlatents'] # mlp, dcgan
config_mnist['nlayers'] = [2,]*config['nlatents']
config_mnist['nfilters'] = [512,256,128,64,32]
config_mnist['filters_size'] = [3,]*config['nlatents']
config_mnist['nonlinearity'] = 'elu' # soft_plus, relu, leaky_relu, tanh
config_mnist['output_layer'] = ['dense',]*config['nlatents'] # dense, conv, conv1x1
config_mnist['upsample'] = False


### SVHN 10 config
config_svhn = {}
# Outputs set up
config_svhn['verbose'] = False
config_svhn['save_every'] = 2000
config_svhn['print_every'] = 200000
config_svhn['save_final'] = True
config_svhn['save_train_data'] = False
config_svhn['vizu_sinkhorn'] = False
config_svhn['vizu_embedded'] = True
config_svhn['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_svhn['vizu_encSigma'] = False
config_svhn['fid'] = False
config_svhn['work_dir'] = 'results_svhn'
config_svhn['plot_num_pics'] = 100
config_svhn['plot_num_cols'] = 10

# Data set up
config_svhn['dataset'] = 'svhn'
config_svhn['data_dir'] = 'svhn'
config_svhn['input_normalize_sym'] = False
config_svhn['SVHN_data_source_url'] = 'http://ufldl.stanford.edu/housenumbers/'

# Experiment set up
config_svhn['train_dataset_size'] = -1
config_svhn['use_extra'] = False
config_svhn['batch_size'] = 128
config_svhn['epoch_num'] = 4120
config_svhn['method'] = 'wae' #vae, wae
config_svhn['use_trained'] = False #train from pre-trained model
config_svhn['e_pretrain'] = False #pretrained the encoder parameters
config_svhn['e_pretrain_sample_size'] = 200
config_svhn['e_pretrain_it'] = 1000

# Opt set up
config_svhn['optimizer'] = 'adam' # adam, sgd
config_svhn['adam_beta1'] = 0.5
config_svhn['lr'] = 0.0002
config_svhn['lr_adv'] = 0.0008
config_svhn['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_svhn['d_norm'] = 'layernorm' #batchnorm, layernorm, none
config_svhn['batch_norm_eps'] = 1e-05
config_svhn['batch_norm_momentum'] = 0.99

# Objective set up
config_svhn['coef_rec'] = 1. # coef recon loss
config_svhn['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_svhn['penalty'] = 'mmd' #sinkhorn, mmd
config_svhn['pen'] = 'wae' #wae, wae_mmd
config_svhn['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_svhn['L'] = 30 #Sinkhorn iteration
config_svhn['mmd_kernel'] = 'IMQ' # RBF, IMQ

# Model set up
config_svhn['nlatents'] = 8
config_svhn['zdim'] = [64,49,36,25,16,9,4,2]
config_svhn['pz_scale'] = 1.
config_svhn['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_svhn['lambda_scalar'] = 10.
config_svhn['lambda'] = [1/config_svhn['zdim'][i] for i in range(config_svhn['nlatents'])]
config_svhn['lambda'].append(0.0001/config_svhn['zdim'][-1])
config_svhn['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_svhn['init_std'] = 0.0099999
config_svhn['init_bias'] = 0.0
config_svhn['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_svhn['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_svhn['filters_size'] = [5,3,3,3,3,3,3,3]
config_svhn['last_archi'] = ['conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','dense']


config_svhn['e_nlatents'] = config_svhn['nlatents'] #config_mnist['nlatents']
config_svhn['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_svhn['e_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, ali, began
config_svhn['e_nlayers'] = [2,2,2,2,2,2,2,2]
config_svhn['e_nfilters'] = [96,96,64,64,32,32,32,32]
config_svhn['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh


config_svhn['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_svhn['d_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, dcgan_mod, ali, began
config_svhn['d_nlayers'] = [2,2,2,2,2,2,2,2]
config_svhn['d_nfilters'] = [96,96,64,64,32,32,32,32]
config_svhn['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


### CIFAR 10 config
config_cifar10 = {}
# Outputs set up
config_cifar10['verbose'] = False
config_cifar10['save_every'] = 2000
config_cifar10['print_every'] = 200000
config_cifar10['save_final'] = True
config_cifar10['save_train_data'] = False
config_cifar10['vizu_sinkhorn'] = False
config_cifar10['vizu_embedded'] = True
config_cifar10['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_cifar10['vizu_encSigma'] = False
config_cifar10['fid'] = False
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
config_cifar10['epoch_num'] = 4120
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
config_cifar10['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_cifar10['d_norm'] = 'layernorm' #batchnorm, layernorm, none
config_cifar10['batch_norm_eps'] = 1e-05
config_cifar10['batch_norm_momentum'] = 0.99

# Objective set up
config_cifar10['coef_rec'] = 1. # coef recon loss
config_cifar10['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_cifar10['penalty'] = 'mmd' #sinkhorn, mmd
config_cifar10['pen'] = 'wae' #wae, wae_mmd
config_cifar10['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_cifar10['L'] = 30 #Sinkhorn iteration
config_cifar10['mmd_kernel'] = 'RQ' # RBF, IMQ, RQ

# Model set up
config_cifar10['nlatents'] = 8
config_cifar10['zdim'] = [64,49,36,25,16,9,4,2]
config_cifar10['pz_scale'] = 1.
config_cifar10['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_cifar10['lambda_scalar'] = 10.
config_cifar10['lambda'] = [1/config_cifar10['zdim'][i] for i in range(config_cifar10['nlatents'])]
config_cifar10['lambda'].append(0.0001/config_cifar10['zdim'][-1])
config_cifar10['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_cifar10['init_std'] = 0.0099999
config_cifar10['init_bias'] = 0.0
config_cifar10['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_cifar10['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_cifar10['filters_size'] = [5,3,3,3,3,3,3,3]
config_cifar10['last_archi'] = ['conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','dense']


config_cifar10['e_nlatents'] = config_cifar10['nlatents'] #config_mnist['nlatents']
config_cifar10['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_cifar10['e_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, ali, began
config_cifar10['e_nlayers'] = [2,2,2,2,2,2,2,2]
config_cifar10['e_nfilters'] = [96,96,64,64,32,32,32,32]
config_cifar10['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh


config_cifar10['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_cifar10['d_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, dcgan_mod, ali, began
config_cifar10['d_nlayers'] = [2,2,2,2,2,2,2,2]
config_cifar10['d_nfilters'] = [96,96,64,64,32,32,32,32]
config_cifar10['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh

### CelebA config
config_celebA = {}
# Outputs set up
config_celebA['verbose'] = False
config_celebA['save_every'] = 2000
config_celebA['print_every'] = 200000
config_celebA['save_final'] = True
config_celebA['save_train_data'] = False
config_celebA['vizu_sinkhorn'] = False
config_celebA['vizu_embedded'] = False
config_celebA['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_celebA['vizu_encSigma'] = False
config_celebA['fid'] = False
config_celebA['work_dir'] = 'results_celebA'
config_celebA['plot_num_pics'] = 100
config_celebA['plot_num_cols'] = 10

# Data set up
config_celebA['dataset'] = 'celebA'
config_celebA['data_dir'] = 'celebA'
config_celebA['input_normalize_sym'] = True
config_celebA['celebA_data_source_url'] = 'https://docs.google.com/uc?export=download'
config_celebA['celebA_crop'] = 'closecrop' # closecrop, resizecrop

# Experiment set up
config_celebA['train_dataset_size'] = -1
config_celebA['batch_size'] = 128
config_celebA['epoch_num'] = 4120
config_celebA['method'] = 'wae' #vae, wae
config_celebA['use_trained'] = False #train from pre-trained model
config_celebA['e_pretrain'] = False #pretrained the encoder parameters
config_celebA['e_pretrain_sample_size'] = 200
config_celebA['e_pretrain_it'] = 1000

# Opt set up
config_celebA['optimizer'] = 'adam' # adam, sgd
config_celebA['adam_beta1'] = 0.5
config_celebA['lr'] = 0.0002
config_celebA['lr_adv'] = 0.0008
config_celebA['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_celebA['d_norm'] = 'layernorm' #batchnorm, layernorm, none
config_celebA['batch_norm_eps'] = 1e-05
config_celebA['batch_norm_momentum'] = 0.99

# Objective set up
config_celebA['coef_rec'] = 1. # coef recon loss
config_celebA['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_celebA['penalty'] = 'mmd' #sinkhorn, mmd
config_celebA['pen'] = 'wae' #wae, wae_mmd
config_celebA['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_celebA['L'] = 30 #Sinkhorn iteration
config_celebA['mmd_kernel'] = 'IMQ' # RBF, IMQ

# Model set up
config_celebA['nlatents'] = 8
config_celebA['zdim'] = [64,49,36,25,16,9,4,2]
config_celebA['pz_scale'] = 1.
config_celebA['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_celebA['lambda_scalar'] = 10.
config_celebA['lambda'] = [1/config_celebA['zdim'][i] for i in range(config_celebA['nlatents'])]
config_celebA['lambda'].append(0.0001/config_celebA['zdim'][-1])
config_celebA['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_celebA['init_std'] = 0.0099999
config_celebA['init_bias'] = 0.0
config_celebA['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_celebA['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_celebA['filters_size'] = [5,3,3,3,3,3,3,3]
config_celebA['last_archi'] = ['conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','dense']


config_celebA['e_nlatents'] = config_celebA['nlatents'] #config_mnist['nlatents']
config_celebA['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_celebA['e_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, ali, began
config_celebA['e_nlayers'] = [2,2,2,2,2,2,2,2]
config_celebA['e_nfilters'] = [96,96,64,64,32,32,32,32]
config_celebA['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh


config_celebA['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_celebA['d_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, dcgan_mod, ali, began
config_celebA['d_nlayers'] = [2,2,2,2,2,2,2,2]
config_celebA['d_nfilters'] = [96,96,64,64,32,32,32,32]
config_celebA['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
