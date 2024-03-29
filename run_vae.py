import os
from datetime import datetime
import sys
import logging
import argparse
import itertools

import tensorflow as tf
from math import pow, sqrt, exp, log

import configs
from train import Run
from datahandler import DataHandler
import utils

import pdb

parser = argparse.ArgumentParser()
# run setup
parser.add_argument(
    "--model", default="vae", help="model to train [vae/wae/lvae/stackedwae]"
)
parser.add_argument("--mode", default="train", help="mode to run [train/vizu/fid/test]")
parser.add_argument(
    "--losses", action="store_false", default=True, help="plot split losses"
)
parser.add_argument(
    "--reconstructions", action="store_false", default=True, help="plot reconstructions"
)
parser.add_argument(
    "--embedded", action="store_false", default=True, help="plot embedded"
)
parser.add_argument(
    "--latents", action="store_false", default=True, help="plot latents expl."
)
parser.add_argument(
    "--fid", action="store_true", default=False, help="compute FID score"
)
parser.add_argument("--dataset", default="mnist", help="dataset")
parser.add_argument(
    "--data_dir", type=str, default="../data", help="directory in which data is stored"
)
parser.add_argument("--num_it", type=int, default=300000, help="iteration number")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate size")
# pretraining
parser.add_argument(
    "--use_trained",
    action="store_true",
    default=False,
    help="whether to use pre trained weights",
)
parser.add_argument("--weights_file")
# path setup
parser.add_argument(
    "--out_dir",
    type=str,
    default="code_outputs",
    help="root_directory in which outputs are saved",
)
parser.add_argument(
    "--res_dir", type=str, default="res", help="directory in which exp. res are saved"
)
# model setup
parser.add_argument("--encoder", type=str, default="gauss", help="encoder type")
parser.add_argument(
    "--sigmoid",
    action="store_false",
    default=True,
    help="use sigmoid activation for det rec.",
)
parser.add_argument(
    "--net_archi",
    type=str,
    default="mlp",
    help="networks architecture [mlp/conv_locatello/conv_rae]",
)
parser.add_argument(
    "--cost",
    type=str,
    default="cross_entropy",
    help="ground cost [l2, l2sq, l2sq_norm, l1, cross_entropy]",
)
parser.add_argument(
    "--lmba_schedule", type=str, default="constant", help="reg schedule"
)
parser.add_argument(
    "--enc_sigma_pen", action="store_true", default=False, help="regularized enc sigma"
)
parser.add_argument(
    "--dec_sigma_pen", action="store_true", default=False, help="regularized enc sigma"
)
# saving setup
parser.add_argument(
    "--save_model",
    action="store_false",
    default=True,
    help="save final model weights [True/False]",
)
parser.add_argument(
    "--save_data", action="store_false", default=True, help="save training data"
)
# exp id
parser.add_argument(
    "--id", type=int, default=1, help="exp id corresponding to latent reg weight setup"
)


FLAGS = parser.parse_args()


def main():

    # Select dataset to use
    if FLAGS.dataset == "mnist":
        opts = configs.config_mnist
    elif FLAGS.dataset == "smallNORB":
        opts = configs.config_smallNORB
    elif FLAGS.dataset == "celebA":
        opts = configs.config_celebA
    else:
        assert False, "Unknown dataset"

    # model
    opts["model"] = FLAGS.model
    opts["encoder"] = [
        FLAGS.encoder,
    ] * opts["nlatents"]
    opts["use_sigmoid"] = FLAGS.sigmoid
    opts["archi"] = [
        FLAGS.net_archi,
    ] * opts["nlatents"]
    opts["obs_cost"] = FLAGS.cost
    opts["lambda_schedule"] = FLAGS.lmba_schedule
    opts["enc_sigma_pen"] = FLAGS.enc_sigma_pen
    opts["dec_sigma_pen"] = FLAGS.dec_sigma_pen

    # opts['nlatents'] = 1
    # zdims = [2,4,8,16]
    # id = (FLAGS.id-1) % len(zdims)
    # opts['zdim'] = [zdims[id],]
    # opts['lambda_init'] = [1,]
    # opts['lambda'] = [1.,]
    # beta = opts['lambda']
    # opts['lambda_sigma'] = [1.,]

    # lamba
    beta = [0.0001, 1.0]
    id = (FLAGS.id - 1) % len(beta)
    opts["lambda_init"] = [beta[id] for n in range(opts["nlatents"])]
    opts["lambda"] = [1.0 for n in range(opts["nlatents"])]

    # Create directories
    results_dir = "results"
    if not tf.io.gfile.isdir(results_dir):
        utils.create_dir(results_dir)
    opts["out_dir"] = os.path.join(results_dir, FLAGS.out_dir)
    if not tf.io.gfile.isdir(opts["out_dir"]):
        utils.create_dir(opts["out_dir"])
    out_subdir = os.path.join(opts["out_dir"], opts["model"])
    if not tf.io.gfile.isdir(out_subdir):
        utils.create_dir(out_subdir)
    # out_subdir = os.path.join(out_subdir, 'dz'+str(zdims[id]))
    # if not tf.io.gfile.isdir(out_subdir):
    #     utils.create_dir(out_subdir)
    opts["exp_dir"] = FLAGS.res_dir
    exp_dir = os.path.join(
        out_subdir,
        "{}_{}layers_lreg{}_{:%Y_%m_%d_%H_%M}".format(
            opts["exp_dir"], opts["nlatents"], beta[id], datetime.now()
        ),
    )
    opts["exp_dir"] = exp_dir
    if not tf.io.gfile.isdir(exp_dir):
        utils.create_dir(exp_dir)
        utils.create_dir(os.path.join(exp_dir, "checkpoints"))

    # getting weights path
    if FLAGS.weights_file is not None:
        WEIGHTS_PATH = os.path.join(opts["exp_dir"], "checkpoints", FLAGS.weights_file)
    else:
        WEIGHTS_PATH = None

    # Verbose
    logging.basicConfig(
        filename=os.path.join(exp_dir, "outputs.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    # run set up
    opts["vizu_splitloss"] = FLAGS.losses
    opts["vizu_fullrec"] = FLAGS.reconstructions
    opts["vizu_embedded"] = FLAGS.embedded
    opts["vizu_latent"] = FLAGS.latents
    opts["fid"] = FLAGS.fid
    opts["it_num"] = FLAGS.num_it
    opts["print_every"] = int(opts["it_num"] / 4)
    opts["evaluate_every"] = int(opts["it_num"] / 50)
    if FLAGS.batch_size is not None:
        opts["batch_size"] = FLAGS.batch_size
    opts["lr"] = FLAGS.lr
    opts["use_trained"] = FLAGS.use_trained
    opts["save_every"] = 10000000000
    opts["save_final"] = FLAGS.save_model
    opts["save_train_data"] = FLAGS.save_data

    # Reset tf graph
    tf.compat.v1.reset_default_graph()

    # Loading the dataset
    opts["data_dir"] = FLAGS.data_dir
    data = DataHandler(opts)
    assert data.train_size >= opts["batch_size"], "Training set too small"

    # build model
    run = Run(opts, data)

    # Training/testing/vizu
    if FLAGS.mode == "train":
        # Dumping all the configs to the text file
        with utils.o_gfile((opts["exp_dir"], "params.txt"), "w") as text:
            text.write("Parameters:\n")
            for key in opts:
                text.write("%s : %s\n" % (key, opts[key]))
        run.train(WEIGHTS_PATH)
    elif FLAGS.mode == "vizu":
        opts["rec_loss_nsamples"] = 1
        opts["sample_recons"] = False
        run.latent_interpolation(opts["exp_dir"], WEIGHTS_PATH)
    elif FLAGS.mode == "fid":
        run.fid_score(WEIGHTS_PATH)
    elif FLAGS.mode == "test":
        run.test_losses(WEIGHTS_PATH)
    elif FLAGS.mode == "vlae_exp":
        run.vlae_experiment(WEIGHTS_PATH)
    else:
        assert False, "Unknown mode %s" % FLAGS.mode


main()
