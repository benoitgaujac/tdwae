import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils
from math import ceil, log

import tensorflow as tf

import pdb

parser = argparse.ArgumentParser()
# Args for experiment

FLAGS = parser.parse_args()

def main():

    # Select dataset to use
    opts = configs.config_mnist

    # training WAE
    opts['method'] = 'wae'
    utils.create_dir(opts['method'])
    root_path = os.path.join(opts['method'],'rand_enc')
    utils.create_dir(root_path)

    # Verbose
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    opts['encoder'] = ['gauss' for i in range(opts['nlatents'])]
    for i in range(opts['nlatents']+1):
        # Create root_directory
        work_dir = 'mnist_%dgauss_%ddet' % (opts['nlatents']-i,i)
        work_path = os.path.join(root_path,work_dir)
        utils.create_dir(work_path)
        opts['work_dir']=work_path
        if i>0:
            opts['encoder'][opts['nlatents']-i]='det'
        # Checkpoints dir
        utils.create_dir(os.path.join(work_path, 'checkpoints'))
        # Dumping all the configs to the text file
        with utils.o_gfile((work_path, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))

        #Reset tf graph
        tf.reset_default_graph()

        # build WAE
        wae = WAE(opts)

        # Training
        wae.train(data, opts['work_dir'], 'none')

main()
