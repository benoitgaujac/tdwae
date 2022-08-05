import sys
from math import sqrt, cos, sin, pi
import numpy as np
import tensorflow as tf

import utils
from sampling_functions import (
    sample_pz,
    sample_gaussian,
    sample_bernoulli,
    sample_unif,
    linespace,
)
from loss_functions import (
    obs_reconstruction_loss,
    latent_penalty,
    latent_reconstruction_loss,
)
from loss_functions import KL, log_normal
from encoder import Encoder
from decoder import Decoder
from datahandler import datashapes

import pdb


class Model(object):
    def __init__(self, opts, pz_params):
        self.opts = opts
        self.pz_params = pz_params

    # --- encode the inputs
    def encode(
        self, inputs, sigma_scale, resample, nresamples=1, reuse=False, is_training=True
    ):
        pass

    # --- encode the inputs
    def decode(self, zs, input_shapes, sigma_scale, latent_id=None, reuse=False, is_training=True):
        pass

    # --- full path through the model
    def forward_pass(
        self, inputs, sigma_scale, resample, nresamples=1, reuse=False, is_training=True
    ):
        # --- encoder-decoder foward pass
        zs, enc_means, enc_Sigmas, self.input_shapes = self.encode(
            inputs, sigma_scale, resample, nresamples, reuse, is_training
        )
        xs, dec_means, dec_Sigmas = self.decode(
            zs, self.input_shapes, sigma_scale, None, reuse, is_training
        )
        return zs, enc_means, enc_Sigmas, xs, dec_means, dec_Sigmas

    # --- reconstruct for each layer
    def reconstruct(self, zs, sigma_scale):
        pass

    # --- sample from the model
    def sample_x_from_prior(self, samples, sigma_scale):
        pass

    # various metrics
    def MSE(self, inputs, sigma_scale):
        # --- compute MSE between inputs and reconstruction
        zs, _, _, input_shapes = self.encode(
            inputs, sigma_scale, False, reuse=True, is_training=False
        )
        xs = self.reconstruct(zs, sigma_scale)
        square_dist = tf.reduce_mean(
            tf.square(tf.compat.v1.layers.flatten(inputs) - xs[-1]), axis=-1
        )
        return tf.reduce_mean(square_dist)

    def inception_Net(self, sess, graph):
        # --- get inception net output
        inception_path = "../inception"
        inception_model = os.path.join(inception_path, "classify_image_graph_def.pb")
        layername = "FID_Inception_Net/pool_3:0"
        # Create inception graph
        with graph.as_default():
            with tf.gfile.FastGFile(inception_model, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name="FID_Inception_Net")
        # Get inception activation layer (and reshape for batching)
        pool3 = sess.graph.get_tensor_by_name(layername)
        ops_pool3 = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops_pool3):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != []:
                    shape = [s.value for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
        return pool3

    def layerwise_kl(self, inputs, sigma_scale):
        # --- compute layer-wise KL(q(z_i|z_i-1,p(z_i|z_i+1))
        _, enc_means, enc_Sigmas, _, dec_means, dec_Sigmas = self.forward_pass(
            inputs, sigma_scale, False, reuse=True, is_training=False
        )
        kls = []
        # latent layer up to N-1
        for n in range(len(enc_means) - 1):
            kl = KL(enc_means[n], enc_Sigmas[n], dec_means[n + 1], dec_Sigmas[n + 1])
            kls.append(kl)
        # deepest layer
        pz_mean, pz_Sigma = tf.split(self.pz_params, 2, axis=-1)
        pz_mean = tf.expand_dims(pz_mean, axis=0)
        pz_Sigma = tf.expand_dims(pz_Sigma, axis=0)
        kl = KL(enc_means[-1], enc_Sigmas[-1], pz_mean, pz_Sigma)
        kls.append(kl)

        return kls

    def layerwise_agg_kl(self, inputs, sigma_scale):
        # --- compute layer-wise KL(q(z_i|z_i-1,p(z_i|z_i+1))
        zs, enc_means, enc_Sigmas, xs, dec_means, dec_Sigmas = self.forward_pass(
            inputs, sigma_scale, False, reuse=True, is_training=False
        )
        kls = []
        # latent layer up to N-1
        for n in range(len(enc_means) - 1):
            logp_prob = log_normal(
                tf.expand_dims(xs[n + 1], 1),
                tf.expand_dims(dec_means[n + 1], 0),
                tf.expand_dims(dec_Sigmas[n + 1], 0),
            )
            logp = tf.reduce_logsumexp(
                tf.reduce_sum(logp_prob, axis=2, keepdims=False), axis=1, keepdims=False
            )
            logq_prob = log_normal(
                tf.expand_dims(zs[n], 1),
                tf.expand_dims(enc_means[n], 0),
                tf.expand_dims(enc_Sigmas[n], 0),
            )
            logq = tf.reduce_logsumexp(
                tf.reduce_sum(logq_prob, axis=2, keepdims=False), axis=1, keepdims=False
            )
            kl = tf.reduce_mean(logp - logq)
            kls.append(kl)
        # deepest layer
        pz_mean, pz_Sigma = tf.split(self.pz_params, 2, axis=-1)
        pz_samples = sample_gaussian(
            self.opts, self.pz_params, "numpy", self.opts["batch_size"]
        )
        logp_prob = log_normal(
            tf.expand_dims(pz_samples, 1),
            tf.expand_dims(tf.expand_dims(pz_mean, axis=0), 0),
            tf.expand_dims(tf.expand_dims(pz_Sigma, axis=0), 0),
        )
        logp = tf.reduce_logsumexp(
            tf.reduce_sum(logp_prob, axis=2, keepdims=False), axis=1, keepdims=False
        )
        logq_prob = log_normal(
            tf.expand_dims(zs[-1], 1),
            tf.expand_dims(enc_means[-1], 0),
            tf.expand_dims(enc_Sigmas[-1], 0),
        )
        logq = tf.reduce_logsumexp(
            tf.reduce_sum(logq_prob, axis=2, keepdims=False), axis=1, keepdims=False
        )
        kl = tf.reduce_mean(logp - logq)
        kls.append(kl)

        return kls


class stackedWAE(Model):
    def __init__(self, opts, pz_params):
        super().__init__(opts, pz_params)

    def encode(
        self, inputs, sigma_scale, resample, nresamples=1, reuse=False, is_training=True
    ):
        # --- Encoding Loop
        zs, means, Sigmas = [], [], []
        input_shapes = []
        input_shape = inputs.get_shape().as_list()[1:]
        for n in range(self.opts["nlatents"]):
            if n == 0:
                input = inputs
            else:
                if resample:
                    # when resampling for vizu, we just pass the mean
                    input = means[-1]
                else:
                    input = zs[-1]
            mean, Sigma, input_shape = Encoder(
                self.opts,
                input=input,
                input_shape=input_shape,
                archi=self.opts["archi"][n],
                nlayers=self.opts["nlayers"][n],
                nfilters=self.opts["nfilters"][n],
                filters_size=self.opts["filters_size"][n],
                output_dim=self.opts["zdim"][n],
                downsample=self.opts["updownsample"][n],
                output_layer=self.opts["last_archi"][n],
                top_latent=(n == self.opts["nlatents"] - 1),
                scope="encoder/layer_%d" % (n + 1),
                reuse=reuse,
                is_training=is_training,
            )
            if self.opts["encoder"][n] == "det":
                # - deterministic encoder
                z = mean
            elif self.opts["encoder"][n] == "gauss":
                # - gaussian encoder
                if resample:
                    q_params = tf.concat((mean, sigma_scale * Sigma), axis=-1)
                    q_params = tf.stack([q_params for i in range(nresamples)], axis=1)
                else:
                    q_params = tf.concat((mean, Sigma), axis=-1)
                z = sample_gaussian(self.opts, q_params, "tensorflow")
            else:
                assert False, "Unknown encoder %s" % self.opts["encoder"]
            zs.append(z)
            means.append(mean)
            Sigmas.append(Sigma)
            input_shapes.append(input_shape)

        return zs, means, Sigmas, input_shapes

    def decode(self, zs, input_shapes, sigma_scale, latent_id=None, reuse=False, is_training=True):
        # --- Decoding Loop
        xs, means, Sigmas = [], [], []
        for n in range(len(zs)):
            if latent_id is not None:
                idx = latent_id
            else:
                idx = n
            if idx == 0:
                output_dim = datashapes[self.opts['dataset']]
            else:
                output_dim = [
                    self.opts["zdim"][idx - 1],
                ]
            z = zs[n]
            zshape = z.get_shape().as_list()[1:]
            if len(zshape) > 1:
                # reshape the codes to [-1,output_dim]
                z = tf.squeeze(tf.concat(tf.split(z, zshape[0], 1), axis=0), [1])
            mean, Sigma = Decoder(
                self.opts,
                input=z,
                input_shape=input_shapes[idx],
                archi=self.opts["archi"][idx],
                nlayers=self.opts["nlayers"][idx],
                nfilters=self.opts["nfilters"][idx],
                filters_size=self.opts["filters_size"][idx],
                output_dim=output_dim,
                upsample=self.opts["updownsample"][idx],
                last_archi=self.opts["last_archi"][idx],
                scope="decoder/layer_%d" % idx,
                reuse=reuse,
                is_training=is_training,
            )
            # reshaping to [-1,nresamples,output_dim] if needed
            if len(zshape) > 1:
                mean = tf.stack(tf.split(mean, zshape[0]), axis=1)
                Sigma = tf.stack(tf.split(Sigma, zshape[0]), axis=1)
            # - resampling reconstruced
            if self.opts["decoder"][idx] == "det":
                # - deterministic decoder
                x = mean
                if self.opts["use_sigmoid"]:
                    x = tf.compat.v1.sigmoid(x)
            elif self.opts["decoder"][idx] == "gauss":
                # - gaussian decoder
                p_params = tf.concat((mean, sigma_scale * Sigma), axis=-1)
                x = sample_gaussian(self.opts, p_params, "tensorflow")
            else:
                assert False, "Unknown encoder %s" % self.opts["decoder"][idx]
            xs.append(x)
            means.append(mean)
            Sigmas.append(Sigma)

        return xs, means, Sigmas

    def losses(self, inputs, sigma_scale, reuse=False, is_training=True):
        # --- compute the losses of the stackedWAE
        zs, enc_means, enc_Sigmas, xs, dec_means, dec_Sigmas = self.forward_pass(
            inputs, sigma_scale, False, reuse=reuse, is_training=is_training
        )
        obs_cost = obs_reconstruction_loss(self.opts, inputs, xs[0])
        latent_cost = self.latent_cost(
            xs[1:],
            dec_means[1:],
            dec_Sigmas[1:],
            zs[:-1],
            enc_means[:-1],
            enc_Sigmas[:-1],
        )
        pz_samples = sample_gaussian(
            self.opts, self.pz_params, "numpy", self.opts["batch_size"]
        )
        # if len(qz_samples.get_Shape().as_list()[1:])>1:
        #     qz_samples = zs[-1][:,0]
        # else:
        #     qz_samples = zs[-1]
        matching_penalty = latent_penalty(self.opts, zs[-1], pz_samples)
        enc_Sigma_penalty = self.Sigma_penalty(enc_Sigmas)
        dec_Sigma_penalty = self.Sigma_penalty(dec_Sigmas[1:])
        return (
            obs_cost,
            latent_cost,
            matching_penalty,
            enc_Sigma_penalty,
            dec_Sigma_penalty,
        )

    def latent_cost(self, xs, x_means, x_Sigmas, zs, z_means, z_Sigmas):
        # --- compute the latent cost for each latent layer last one
        costs = []
        for n in range(len(zs)):
            costs.append(
                latent_reconstruction_loss(
                    self.opts,
                    xs[n],
                    x_means[n],
                    x_Sigmas[n],
                    zs[n],
                    z_means[n],
                    z_Sigmas[n],
                )
            )
        return costs

    def Sigma_penalty(self, Sigmas):
        # -- compute the encoder Sigmas penalty
        penalties = []
        for n in range(len(Sigmas)):
            penalty = tf.reduce_mean(
                tf.abs(tf.reduce_sum(tf.compat.v1.log(Sigmas[n]), axis=-1))
            )
            penalties.append(penalty)
        return penalties

    def reconstruct(self, zs, sigma_scale):
        # Reconstruct for each encoding layer
        reconstruction = []
        inputs = zs
        for m in range(len(zs)):
            rec, _, _ = self.decode(inputs, self.input_shapes, sigma_scale, None, True, False)
            reconstruction.append(rec[0])
            inputs = rec[1:]

        return reconstruction

    def sample_x_from_prior(self, samples, sigma_scale):
        # --- sample from prior noise
        decoded, means, Sigmas = [], [], []
        inputs = samples
        for m in range(self.opts["nlatents"]):
            dec, mean, Sigma = self.decode(
                [
                    inputs,
                ],
                self.input_shapes,
                sigma_scale,
                self.opts["nlatents"] - (m + 1),
                True,
                False,
            )
            decoded.append(dec[0])
            means.append(mean[0])
            Sigmas.append(Sigma[0])
            inputs = dec[0]

        return decoded, means, Sigmas


class WAE(Model):
    def __init__(self, opts, pz_params):
        super().__init__(opts, pz_params)

    def encode(
        self, inputs, sigma_scale, resample, nresamples=1, reuse=False, is_training=True
    ):
        # --- Encoding Loop
        mean, Sigma = Encoder(
            self.opts,
            input=inputs,
            archi=self.opts["archi"][0],
            nlayers=self.opts["nlayers"][0],
            nfilters=self.opts["nfilters"][0],
            filters_size=self.opts["filters_size"][0],
            output_dim=self.opts["zdim"][0],
            downsample=self.opts["upsample"],
            output_layer=self.opts["output_layer"][0],
            scope="encoder/layer_1",
            reuse=reuse,
            is_training=is_training,
        )
        if self.opts["encoder"][0] == "det":
            # - deterministic encoder
            z = mean
        elif self.opts["encoder"][0] == "gauss":
            # - gaussian encoder
            if resample:
                q_params = tf.concat((mean, sigma_scale * Sigma), axis=-1)
                q_params = tf.stack([q_params for i in range(nresamples)], axis=1)
            else:
                q_params = tf.concat((mean, Sigma), axis=-1)
            z = sample_gaussian(self.opts, q_params, "tensorflow")
        else:
            assert False, "Unknown encoder %s" % self.opts["encoder"]

        return (
            [
                z,
            ],
            [
                mean,
            ],
            [
                Sigma,
            ],
        )

    def decode(self, zs, sigma_scale, latent_id=None, reuse=False, is_training=True):
        # --- Decoding Loop
        output_dim = datashapes[self.opts["dataset"]][:-1] + [
            datashapes[self.opts["dataset"]][-1],
        ]
        z = zs[0]
        zshape = z.get_shape().as_list()[1:]
        if len(zshape) > 1:
            # reshape the codes to [-1,output_dim]
            z = tf.squeeze(tf.concat(tf.split(z, zshape[0], 1), axis=0), [1])
        mean, Sigma = Decoder(
            self.opts,
            input=z,
            archi=self.opts["archi"][0],
            nlayers=self.opts["nlayers"][0],
            nfilters=self.opts["nfilters"][0],
            filters_size=self.opts["filters_size"][0],
            output_dim=output_dim,
            upsample=self.opts["upsample"],
            output_layer=self.opts["output_layer"][0],
            scope="decoder/layer_0",
            reuse=reuse,
            is_training=is_training,
        )
        # reshaping to [-1,nresamples,output_dim] if needed
        if len(zshape) > 1:
            mean = tf.stack(tf.split(mean, zshape[0]), axis=1)
            Sigma = tf.stack(tf.split(Sigma, zshape[0]), axis=1)
        # - resampling reconstruced
        if self.opts["decoder"][0] == "det":
            # - deterministic decoder
            x = mean
            if self.opts["use_sigmoid"]:
                x = tf.compat.v1.sigmoid(x)
        elif self.opts["decoder"][0] == "gauss":
            # - gaussian decoder
            p_params = tf.concat((mean, sigma_scale * Sigma), axis=-1)
            x = sample_gaussian(self.opts, p_params, "tensorflow")
        else:
            assert False, "Unknown encoder %s" % self.opts["decoder"][idx]

        return (
            [
                x,
            ],
            [
                mean,
            ],
            [
                Sigma,
            ],
        )

    def decode_implicit_prior(self, z, reuse=False, is_training=True):
        # --- Decoding Loop
        output = z
        for n in range(self.opts["nlatents"] - 1, 0, -1):
            if n == 1:
                output_dim = [
                    self.opts["zdim"][n - 1],
                ]
            else:
                output_dim = None
            output, _ = Decoder(
                self.opts,
                input=output,
                archi=self.opts["archi"][n],
                nlayers=self.opts["nlayers"][n],
                nfilters=self.opts["nfilters"][n],
                filters_size=self.opts["filters_size"][n],
                output_dim=output_dim,
                upsample=self.opts["upsample"],
                output_layer=self.opts["output_layer"][n],
                scope="decoder/layer_%d" % n,
                reuse=reuse,
                is_training=is_training,
            )

        return output

    def losses(
        self, inputs, sigma_scale, resample, nresamples=1, reuse=False, is_training=True
    ):
        # --- compute the losses of the stackedWAE
        zs, _, enc_Sigmas, xs, _, _ = self.forward_pass(
            inputs, sigma_scale, resample, nresamples, reuse, is_training
        )
        obs_cost = obs_reconstruction_loss(self.opts, inputs, xs[0])
        latent_cost = []
        pz_samples = tf.convert_to_tensor(
            sample_gaussian(self.opts, self.pz_params, "numpy", self.opts["batch_size"])
        )
        pz_samples = self.decode_implicit_prior(pz_samples, reuse, is_training)
        if resample:
            qz_samples = zs[-1][:, 0]
        else:
            qz_samples = zs[-1]
        matching_penalty = latent_penalty(self.opts, qz_samples, pz_samples)
        Sigma_penalty = self.Sigma_penalty(enc_Sigmas)
        return obs_cost, latent_cost, matching_penalty, Sigma_penalty

    def Sigma_penalty(self, Sigmas):
        # -- compute the encoder Sigmas penalty
        penalties = []
        for n in range(len(Sigmas)):
            penalty = tf.reduce_mean(
                tf.abs(tf.reduce_mean(tf.compat.v1.log(Sigmas[n]), axis=-1))
            )
            penalties.append(penalty)
        return penalties

    def reconstruct(self, zs, sigma_scale):
        # Reconstruct for each encoding layer
        reconstruction = []
        inputs = zs
        for m in range(len(zs)):
            rec, _, _ = self.decode(inputs, sigma_scale, None, True, False)
            reconstruction.append(rec[0])
            inputs = rec[1:]

        return reconstruction

    def sample_x_from_prior(self, samples, sigma_scale):
        # --- sample from prior noise
        outputs = self.decode_implicit_prior(samples, True, False)
        decoded, means, Sigmas = self.decode(
            [
                outputs,
            ],
            sigma_scale,
            reuse=True,
            is_training=False,
        )

        return decoded, means, Sigmas


class VAE(Model):
    def __init__(self, opts, pz_params):
        super().__init__(opts, pz_params)

    def encode(
        self, inputs, sigma_scale, resample, nresamples=1, reuse=False, is_training=True
    ):
        # --- Encoding Loop
        zs, means, Sigmas = [], [], []
        for n in range(self.opts["nlatents"]):
            if n == 0:
                input = inputs
            else:
                if resample:
                    # when resampling for vizu, we just pass the mean
                    input = means[-1]
                else:
                    input = zs[-1]
            mean, Sigma = Encoder(
                self.opts,
                input=input,
                archi=self.opts["archi"][n],
                nlayers=self.opts["nlayers"][n],
                nfilters=self.opts["nfilters"][n],
                filters_size=self.opts["filters_size"][n],
                output_dim=self.opts["zdim"][n],
                downsample=self.opts["upsample"],
                output_layer=self.opts["output_layer"][n],
                scope="encoder/layer_%d" % (n + 1),
                reuse=reuse,
                is_training=is_training,
            )
            if self.opts["encoder"][n] == "det":
                # - deterministic encoder
                z = mean
            elif self.opts["encoder"][n] == "gauss":
                # - gaussian encoder
                if resample:
                    q_params = tf.concat((mean, sigma_scale * Sigma), axis=-1)
                    q_params = tf.stack([q_params for i in range(nresamples)], axis=1)
                else:
                    q_params = tf.concat((mean, Sigma), axis=-1)
                z = sample_gaussian(self.opts, q_params, "tensorflow")
            else:
                assert False, "Unknown encoder %s" % self.opts["encoder"]
            zs.append(z)
            means.append(mean)
            Sigmas.append(Sigma)

        return zs, means, Sigmas

    def decode(self, zs, sigma_scale, latent_id=None, reuse=False, is_training=True):
        # --- Decoding Loop
        xs, means, Sigmas = [], [], []
        for n in range(len(zs)):
            if latent_id is not None:
                idx = latent_id
            else:
                idx = n
            if idx == 0:
                output_dim = datashapes[self.opts["dataset"]][:-1] + [
                    datashapes[self.opts["dataset"]][-1],
                ]
            else:
                output_dim = [
                    self.opts["zdim"][idx - 1],
                ]
            z = zs[n]
            zshape = z.get_shape().as_list()[1:]
            if len(zshape) > 1:
                # reshape the codes to [-1,output_dim]
                z = tf.squeeze(tf.concat(tf.split(z, zshape[0], 1), axis=0), [1])
            mean, Sigma = Decoder(
                self.opts,
                input=z,
                archi=self.opts["archi"][idx],
                nlayers=self.opts["nlayers"][idx],
                nfilters=self.opts["nfilters"][idx],
                filters_size=self.opts["filters_size"][idx],
                output_dim=output_dim,
                # features_dim=features_dim,
                upsample=self.opts["upsample"],
                output_layer=self.opts["output_layer"][idx],
                scope="decoder/layer_%d" % idx,
                reuse=reuse,
                is_training=is_training,
            )
            # reshaping to [-1,nresamples,output_dim] if needed
            if len(zshape) > 1:
                mean = tf.stack(tf.split(mean, zshape[0]), axis=1)
                Sigma = tf.stack(tf.split(Sigma, zshape[0]), axis=1)
            # - resampling reconstruced
            if self.opts["decoder"][idx] == "det":
                # - deterministic decoder
                x = mean
                if self.opts["use_sigmoid"]:
                    x = tf.compat.v1.sigmoid(x)
            elif self.opts["decoder"][idx] == "gauss":
                # - gaussian decoder
                p_params = tf.concat((mean, sigma_scale * Sigma), axis=-1)
                x = sample_gaussian(self.opts, p_params, "tensorflow")
            elif self.opts["decoder"][idx] == "bernoulli":
                x = sample_bernoulli(mean)
            else:
                assert False, "Unknown encoder %s" % self.opts["decoder"][idx]
            xs.append(x)
            means.append(mean)
            Sigmas.append(Sigma)

        return xs, means, Sigmas

    def losses(self, inputs, sigma_scale, reuse=False, is_training=True):
        # --- compute the losses of the stackedWAE
        zs, enc_means, enc_Sigmas, xs, dec_means, dec_Sigmas = self.forward_pass(
            inputs, sigma_scale, False, reuse=reuse, is_training=is_training
        )
        # obs
        obs_cost = obs_reconstruction_loss(self.opts, inputs, xs[0])
        # latent
        latent_cost = self.latent_cost(
            xs[1:],
            dec_means[1:],
            dec_Sigmas[1:],
            zs[:-1],
            enc_means[:-1],
            enc_Sigmas[:-1],
        )
        # match
        pz_mean, pz_Sigma = np.split(self.pz_params, 2, axis=-1)
        pz_mean = tf.expand_dims(pz_mean, axis=0)
        pz_Sigma = tf.expand_dims(pz_Sigma, axis=0)
        matching_penalty = KL(enc_means[-1], enc_Sigmas[-1], pz_mean, pz_Sigma)
        # sigma
        enc_Sigma_penalty = self.Sigma_penalty(enc_Sigmas)
        dec_Sigma_penalty = self.Sigma_penalty(dec_Sigmas[1:])
        return (
            obs_cost,
            latent_cost,
            matching_penalty,
            enc_Sigma_penalty,
            dec_Sigma_penalty,
        )

    def latent_cost(self, xs, x_means, x_Sigmas, zs, z_means, z_Sigmas):
        # --- compute the latent cost for each latent layer last one
        costs = []
        for n in range(len(xs)):
            kl = KL(z_means[n], z_Sigmas[n], x_means[n], x_Sigmas[n])
            costs.append(kl)
        return costs

    def Sigma_penalty(self, Sigmas):
        # -- compute the encoder Sigmas penalty
        penalties = []
        for n in range(len(Sigmas)):
            penalty = tf.reduce_mean(
                tf.abs(tf.reduce_sum(tf.compat.v1.log(Sigmas[n]), axis=-1) - 1.0)
            )
            penalties.append(penalty)
        return penalties

    def reconstruct(self, zs, sigma_scale):
        # Reconstruct for each encoding layer
        reconstruction = []
        inputs = zs
        input_shapes = self.input_shapes
        for m in range(len(zs)):
            rec, _, _ = self.decode(inputs, input_shapes, sigma_scale, None, True, False)
            reconstruction.append(rec[0])
            inputs = rec[1:]
            input_shapes = input_shapes[:-1]

        return reconstruction

    def sample_x_from_prior(self, samples, sigma_scale):
        # --- sample from prior noise
        decoded, means, Sigmas = [], [], []
        inputs = samples
        for m in range(self.opts["nlatents"]):
            dec, mean, Sigma = self.decode(
                [
                    inputs,
                ],
                [self.input_shapes[self.opts["nlatents"] - (m + 1)],],
                sigma_scale,
                self.opts["nlatents"] - (m + 1),
                True,
                False,
            )
            decoded.append(dec[0])
            means.append(mean[0])
            Sigmas.append(Sigma[0])
            inputs = dec[0]

        return decoded, means, Sigmas
