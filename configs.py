import copy
from math import pow, sqrt

### MNIST config
config_mnist = {}
# Outputs set up
config_mnist['verbose'] = False
config_mnist['save_every_epoch'] = 10000
config_mnist['print_every'] = 40000
config_mnist['vizu_sinkhorn'] = False
config_mnist['vizu_embedded'] = True
config_mnist['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_mnist['vizu_encSigma'] = True
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
config_mnist['batch_size'] = 256
config_mnist['epoch_num'] = 2000
config_mnist['method'] = 'wae' #vae, wae
config_mnist['use_trained'] = False #train from pre-trained model
config_mnist['e_pretrain'] = False #pretrained the encoder parameters
config_mnist['e_pretrain_it'] = 1000
config_mnist['e_pretrain_sample_size'] = 200

# Opt set up
config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 0.0005
config_mnist['lr_adv'] = 0.0008
config_mnist['batch_norm'] = True
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_momentum'] = 0.99

# Objective set up
config_mnist['coef_rec'] = 1. # coef recon loss
config_mnist['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_mnist['penalty'] = 'mmd' #sinkhorn, mmd
config_mnist['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_mnist['L'] = 30 #Sinkhorn iteration
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ

# Model set up
config_mnist['nlatents'] = 6
config_mnist['zdim'] = [64,32,16,8,4,2]
config_mnist['pz_scale'] = 1.
config_mnist['prior'] = 'gaussian' # dirichlet, gaussian or implicit

# lambda set up
config_mnist['lambda_scalar'] = 10.
config_mnist['lambda'] = [1. for i in range(len(config_mnist['zdim'])-1)]
config_mnist['lambda'].append(config_mnist['lambda_scalar'])
# config_mnist['lambda'] = [config_mnist['zdim'][i]*config_mnist['lambda_scalar']**(i+1)/784 for i in range(len(config_mnist['zdim'])-1)]
# config_mnist['lambda'].append(config_mnist['coef_rec']*config_mnist['lambda_scalar']**config_mnist['nlatents']/784)
config_mnist['lambda_schedule'] = 'adaptive' # adaptive, constant

# NN set up
config_mnist['filter_size'] = 3
config_mnist['init_std'] = 0.0099999
config_mnist['init_bias'] = 0.0
config_mnist['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_mnist['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm

config_mnist['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_mnist['e_arch'] = ['mlp','mlp','mlp','mlp','mlp','mlp'] # mlp, dcgan
config_mnist['e_nlayers'] = [2,2,2,2,2,2]
config_mnist['e_nfilters'] = [1024,512,256,128,64,32]
config_mnist['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh

config_mnist['decoder'] = ['det','det','det','det','det','det'] # deterministic, gaussian
config_mnist['d_arch'] = ['mlp','mlp','mlp','mlp','mlp','mlp'] # mlp, dcgan, dcgan_mod/
config_mnist['d_nlayers'] = [2,2,2,2,2,2]
config_mnist['d_nfilters'] = [1024,512,256,128,64,32]
config_mnist['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


### CIFAR 10 config
config_cifar10 = {}
# Outputs set up
config_cifar10['verbose'] = False
config_cifar10['save_every_epoch'] = 1000
config_cifar10['print_every'] = 5
config_cifar10['vizu_sinkhorn'] = False
config_cifar10['vizu_embedded'] = True
config_cifar10['embedding'] = 'pca' #vizualisation method of the embeddings: pca, umap
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
config_cifar10['batch_size'] = 64
config_cifar10['epoch_num'] = 10
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
config_cifar10['batch_norm_momentum'] = 0.99

# Objective set up
config_cifar10['coef_rec'] = 1. # coef recon loss
config_cifar10['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_cifar10['penalty'] = 'mmd' #sinkhorn, mmd
config_cifar10['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_cifar10['L'] = 30 #Sinkhorn iteration
config_cifar10['mmd_kernel'] = 'IMQ' # RBF, IMQ

# Model set up
config_cifar10['nlatents'] = 1
config_cifar10['zdim'] = [128,] #[64,16,8]
config_cifar10['pz_scale'] = 1.
config_cifar10['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_cifar10['lambda_scalar'] = 10.
config_cifar10['lambda'] = [1. for i in range(len(config_cifar10['zdim'])-1)]
config_cifar10['lambda'].append(config_cifar10['lambda_scalar'])
config_cifar10['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_cifar10['filter_size'] = 3
config_cifar10['init_std'] = 0.099999
config_cifar10['init_bias'] = 0.0
config_cifar10['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_cifar10['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm

config_cifar10['encoder'] = ['det',]#['gauss','gauss','gauss'] # deterministic, gaussian
config_cifar10['e_arch'] = ['resnet',] #['dcgan','mlp','mlp','mlp','mlp'] # mlp, dcgan, ali, began
config_cifar10['e_nlayers'] = [2,2,2,2,2]
config_cifar10['e_nfilters'] = [64,32,16]
config_cifar10['e_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh


config_cifar10['decoder'] = ['det',] #['det','det','det'] # deterministic, gaussian
config_cifar10['d_arch'] = ['resnet',]#['dcgan','mlp','mlp','mlp','mlp'] # mlp, dcgan, dcgan_mod, ali, began
config_cifar10['d_nlayers'] = [2,2,2,2,2]
config_cifar10['d_nfilters'] = [64,32,16]
config_cifar10['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh

### celeba config
config_celeba = {}
# Outputs set up
config_celeba['verbose'] = False
config_celeba['save_every_epoch'] = 1000
config_celeba['print_every'] = 5
config_celeba['vizu_sinkhorn'] = False
config_celeba['vizu_embedded'] = True
config_celeba['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_celeba['work_dir'] = 'results_celeba'
config_celeba['result_dir'] = '../results_celeba'
config_celeba['plot_num_pics'] = 100
config_celeba['plot_num_cols'] = 10

# Data set up
config_celeba['dataset'] = 'celebA'
config_celeba['data_dir'] = 'CelebA/images'
config_celeba['celebA_crop'] = 'closecrop' # closecrop, resizecrop
config_celeba['input_normalize_sym'] = False
config_celeba['cifar10_data_source_url'] = 'https://www.cs.toronto.edu/~kriz/'

# Experiment set up
config_celeba['train_dataset_size'] = -1
config_celeba['batch_size'] = 64
config_celeba['epoch_num'] = 10
config_celeba['method'] = 'wae' #vae, wae
config_celeba['use_trained'] = False #train from pre-trained model
config_celeba['e_pretrain'] = False #pretrained the encoder parameters
config_celeba['e_pretrain_sample_size'] = 200
config_celeba['e_pretrain_it'] = 1000

# Opt set up
config_celeba['optimizer'] = 'adam' # adam, sgd
config_celeba['adam_beta1'] = 0.5
config_celeba['lr'] = 0.0001
config_celeba['lr_adv'] = 0.0008
config_celeba['batch_norm'] = True
config_celeba['batch_norm_eps'] = 1e-05
config_celeba['batch_norm_momentum'] = 0.99

# Objective set up
config_celeba['coef_rec'] = 1. # coef recon loss
config_celeba['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_celeba['penalty'] = 'mmd' #sinkhorn, mmd
config_celeba['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_celeba['L'] = 30 #Sinkhorn iteration
config_celeba['mmd_kernel'] = 'IMQ' # RBF, IMQ

# Model set up
config_celeba['nlatents'] = 4
config_celeba['zdim'] = [1024,256,64,32] #[64,16,8]
config_celeba['pz_scale'] = 1.
config_celeba['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_celeba['lambda_scalar'] = 10.
config_celeba['lambda'] = [1. for i in range(len(config_celeba['zdim'])-1)]
config_celeba['lambda'].append(config_celeba['lambda_scalar'])
config_celeba['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_celeba['filter_size'] = 3
config_celeba['init_std'] = 0.099999
config_celeba['init_bias'] = 0.0
config_celeba['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_celeba['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm

config_celeba['encoder'] = ['gauss']*config_celeba['nlatents']   #['gauss','gauss','gauss'] # deterministic, gaussian
config_celeba['e_arch'] = ['dcgan']*config_celeba['nlatents']  #['dcgan','mlp','mlp','mlp','mlp'] # mlp, dcgan, ali, began
config_celeba['e_nlayers'] = [2]*config_celeba['nlatents']
config_celeba['e_nfilters'] = [1024,512,128,32]
config_celeba['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh


config_celeba['decoder'] = ['det']*config_celeba['nlatents']  #['det','det','det'] # deterministic, gaussian
config_celeba['d_arch'] = ['dcgan']*config_celeba['nlatents'] #['dcgan','mlp','mlp','mlp','mlp'] # mlp, dcgan, dcgan_mod, ali, began
config_celeba['d_nlayers'] = [2]*config_celeba['nlatents']
config_celeba['d_nfilters'] = [1024,512,128,32]
config_celeba['d_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh
