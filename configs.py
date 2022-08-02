import copy
from math import pow, sqrt, exp
import numpy as np

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
config['embedding'] = 'pca' #vizualisation method of the embeddings: pca, umap
config['vizu_latent'] = True
config['vizu_pz_grid'] = True
config['vizu_stochasticity'] = True
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
config['obs_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1, cross_entropy
config['latent_cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l2sq_gauss, l1
config['penalty'] = 'mmd' #sinkhorn, mmd
config['mmd_kernel'] = 'IMQ' # RBF, IMQ
config['sqrdist'] = 'broadcast' #dotprod, broadcast

# Model set up
config['nlatents'] = 5
config['zdim'] = [32,16,8,4,2]
config['pz_scale'] = 1.
config['sigma_scale_resample'] = np.ones(1)
config['sigma_scale_stochasticity'] = [np.ones(1),]
config['prior'] = 'gaussian' # dirichlet, gaussian
config['encoder'] = ['gauss',]*config['nlatents'] # det, gaussian
config['decoder'] = ['det',]+['gauss',]*(config['nlatents']-1) # det, gaussian
config['resamples'] = False
config['nresamples'] = 1

# lambda set up
lrec = 0.0001
config['lambda'] = [lrec**n/config['zdim'][n] for n in range(config['nlatents'])]
config['lambda'].append(lrec**5/config['zdim'][-1])
lrec /= 100
config['lambda_init'] = [lrec**n/config['zdim'][n] for n in range(config['nlatents'])]
config['lambda_init'].append(lrec**5/config['zdim'][-1])
config['lambda_schedule'] = 'constant' # adaptive, constant

# Sigma penalties
config['enc_sigma_pen'] = False # True, False
config['dec_sigma_pen'] = False # True, False
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

# Experiment set up
config_mnist['batch_size'] = 128

# Model set up
config_mnist['nlatents'] = 5
config_mnist['zdim'] = [32,16,8,4,2]
config_mnist['sigma_scale_resample'] = 16.*np.ones(1)
config_mnist['sigma_scale_stochasticity'] = [eps**2*np.ones(1) for eps in [0.01,0.05,0.1,0.5,1.,2.]]
config_mnist['resample'] = True
config_mnist['nresamples'] = 9

# lambda set up
config_mnist['lambda'] = [0.001**n for n in range(1,config_mnist['nlatents']+1)]
config_mnist['lambda_schedule'] = 'constant' # adaptive, constant

# Sigma penalties
config_mnist['enc_sigma_pen'] = False # True, False
config_mnist['dec_sigma_pen'] = False # True, False
config_mnist['lambda_sigma'] = [exp(1-i) for i in range(config_mnist['nlatents'])]

# NN set up
config_mnist['archi'] = ['mlp',]*config['nlatents'] # mlp, dcgan
config_mnist['nlayers'] = [2,]*config['nlatents']
config_mnist['nfilters'] = [2048,1024,512,256,128]
config_mnist['filters_size'] = [3,]*config['nlatents']
config_mnist['nonlinearity'] = 'elu' # soft_plus, relu, leaky_relu, tanh
config_mnist['output_layer'] = ['mlp',]*config['nlatents'] # dense, conv, conv1x1
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
config_svhn['mlpinit'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_svhn['convinit'] = 'he' #he, glorot, normilized_glorot, truncated_norm
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


### CelebA config
config_celebA = config.copy()
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

# Data set up
config_celebA['dataset'] = 'celebA'
config_celebA['data_dir'] = 'celebA'
config_celebA['input_normalize_sym'] = True
config_celebA['celebA_data_source_url'] = 'https://docs.google.com/uc?export=download'
config_celebA['dataset_size'] = 202599
config_celebA['crop_style'] = 'closecrop' # closecrop, resizecrop

# Experiment set up
config_celebA['batch_size'] = 128
config_celebA['train_dataset_size'] = -1
config_celebA['use_trained'] = False #train from pre-trained model

# Opt set up
config_celebA['optimizer'] = 'adam' # adam, sgd
config_celebA['adam_beta1'] = 0.5
config_celebA['lr'] = 0.0002
config_celebA['lr_adv'] = 0.0008
config_celebA['batch_norm_eps'] = 1e-05
config_celebA['batch_norm_momentum'] = 0.99

# Model set up
config_celebA['nlatents'] = 2 #8
config_celebA['zdim'] = [4,2] #[64,49,36,25,16,9,4,2]

# Sigma setup
config_celebA['sigma_scale_resample'] = 16.*np.ones(1)
config_celebA['sigma_scale_stochasticity'] = [eps**2*np.ones(1) for eps in [0.01,0.05,0.1,0.5,1.,2.]]
config_celebA['resample'] = True
config_celebA['nresamples'] = 9
config_celebA['enc_sigma_pen'] = False # True, False
config_celebA['dec_sigma_pen'] = False # True, False
config_celebA['lambda_sigma'] = [exp(1-i) for i in range(config_celebA['nlatents'])]

# lambda set up
config_celebA['lambda_scalar'] = 10.
config_celebA['lambda'] = [1/config_celebA['zdim'][i] for i in range(config_celebA['nlatents'])]
config_celebA['lambda'].append(0.0001/config_celebA['zdim'][-1])
config_celebA['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_celebA['init_std'] = 0.0099999
config_celebA['init_bias'] = 0.0
config_celebA['mlpinit'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_celebA['convinit'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_celebA['encoder'] = ['gauss']*config_celebA['nlatents'] # deterministic, gaussian
config_celebA['decoder'] = ['det',]+['gauss',]*(config_celebA['nlatents']-1) # deterministic, gaussian
config_celebA['arch'] = ['dcgan']*config_celebA['nlatents'] # mlp, dcgan, ali, began
config_celebA['last_archi'] = ['conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','dense']
config_celebA['nlayers'] = [2,2,2,2,2,2,2,2]
config_celebA['filters_size'] = [5,3,3,3,3,3,3,3]
config_celebA['nfilters'] = [96,96,64,64,32,32,32,32]
config_celebA['nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
config_celebA['output_layer'] = ['mlp',]*config['nlatents'] # dense, conv, conv1x1
config_celebA['upsample'] = True
