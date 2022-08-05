# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os

# import shutil
import random
import logging
import gzip
import zipfile
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import numpy as np
from six.moves import cPickle
import urllib.request
import requests
from scipy.io import loadmat
from sklearn.feature_extraction import image
import struct
from tqdm import tqdm
from PIL import Image
import sys
import tarfile
import h5py
from math import ceil

import utils

import pdb

datashapes = {}
datashapes["mnist"] = [28, 28, 1]
datashapes["smallNORB"] = [64, 64, 1]
datashapes["celebA"] = [64, 64, 3]


def _data_dir(opts):
    data_path = maybe_download(opts)
    return data_path


def maybe_download(opts):
    """Download the data from url, unless it's already here."""
    if not tf.io.gfile.exists(opts["data_dir"]):
        tf.io.gfile.makedirs(opts["data_dir"])
    data_path = os.path.join(opts["data_dir"], opts["dataset"])
    if not tf.io.gfile.exists(data_path):
        tf.io.gfile.makedirs(data_path)
    if opts["dataset"] == "mnist":
        files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]
        for filename in files:
            file_path = os.path.join(data_path, filename)
            if not tf.io.gfile.exists(file_path):
                download_file(data_path, file_path, opts["MNIST_data_source_url"])
    elif opts["dataset"] == "smallNORB":
        filename = "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz"
        file_path = os.path.join(data_path, filename)
        if not tf.io.gfile.exists(file_path):
            download_file(file_path, filename, opts["smallNORB_data_source_url"])
        filename = "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz"
        file_path = os.path.join(data_path, filename)
        if not tf.io.gfile.exists(file_path):
            download_file(file_path, filename, opts["smallNORB_data_source_url"])
    elif opts["dataset"] == "celebA":
        filename = "img_align_celeba"
        file_path = os.path.join(data_path, filename)
        if not tf.io.gfile.exists(file_path):
            filename = "img_align_celeba.zip"
            file_path = os.path.join(data_path, filename)
            if not tf.io.gfile.exists(file_path):
                assert False, "{} dataset does not exist".format(opts["dataset"])
                download_file_from_google_drive(
                    file_path, filename, opts["celebA_data_source_url"]
                )
            # Unzipping
            print("Unzipping celebA...")
            with zipfile.ZipFile(file_path) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(data_path)
            print("Unzipping done.")
            os.remove(file_path)
    else:
        assert False, "Unknow dataset"

    return data_path


def download_file(file_path, filename, url):
    file_path, _ = urllib.request.urlretrieve(url + filename, file_path)
    with tf.gfile.GFile(file_path) as f:
        size = f.size()
    print("Successfully downloaded", filename, size, "bytes.")


def download_file_from_google_drive(file_path, filename, url):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    session = requests.Session()
    id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    response = session.get(url, params={"id": id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(url, params=params, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(file_path, "wb") as f:
        for chunk in tqdm(
            response.iter_content(32 * 1024),
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=file_path,
        ):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def load_cifar_batch(fpath, label_key="labels"):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    f = utils.o_gfile(fpath, "rb")
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding="bytes")
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def matrix_type_from_magic(magic_number):
    """
    Get matrix data type from magic number
    See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.
    Parameters
    ----------
    magic_number: tuple
        First 4 bytes read from small NORB files
    Returns
    -------
    element type of the matrix
    """
    convention = {
        "1E3D4C51": "single precision matrix",
        "1E3D4C52": "packed matrix",
        "1E3D4C53": "double precision matrix",
        "1E3D4C54": "integer matrix",
        "1E3D4C55": "byte matrix",
        "1E3D4C56": "short matrix",
    }
    magic_str = bytearray(reversed(magic_number)).hex().upper()
    return convention[magic_str]


def _parse_smallNORB_header(file_pointer):
    """
    Parse header of small NORB binary file

    Parameters
    ----------
    file_pointer: BufferedReader
        File pointer just opened in a small NORB binary file
    Returns
    -------
    file_header_data: dict
        Dictionary containing header information
    """
    # Read magic number
    magic = struct.unpack("<BBBB", file_pointer.read(4))  # '<' is little endian)

    # Read dimensions
    dimensions = []
    (num_dims,) = struct.unpack("<i", file_pointer.read(4))  # '<' is little endian)
    for _ in range(num_dims):
        dimensions.extend(struct.unpack("<i", file_pointer.read(4)))

    file_header_data = {
        "magic_number": magic,
        "matrix_type": matrix_type_from_magic(magic),
        "dimensions": dimensions,
    }
    return file_header_data


def _read_binary_matrix(filename):
    """Reads and returns binary formatted matrix stored in filename."""
    with tf.gfile.GFile(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])

        dtype_map = {
            507333717: "int8",
            507333716: "int32",
            507333713: "float",
            507333715: "double",
        }
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data


def _resize_images(integer_images):
    resized_images = np.zeros((integer_images.shape[0], 64, 64))
    for i in range(integer_images.shape[0]):
        image = Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images.astype(np.float32) / 255.0


class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """

    def __init__(self, opts, seed=None):
        self.dataset = opts["dataset"]
        self.crop_style = opts["crop_style"]
        # load data
        logging.error("\n Loading {}.".format(self.dataset))
        self._create_tfdataset(opts, seed)
        logging.error("Loading Done.")

    def _create_tfdataset(self, opts, seed=None):
        """Crete tfdataset and fill all the necessary variables."""

        # load data
        if self.dataset == "mnist":
            self.data, labels = self._load_mnist(opts)
        elif self.dataset == "smallNORB":
            self.data, labels = self._load_smallNORB(opts)
        elif self.dataset == "celebA":
            self.data, labels = self._load_celebA(opts)
        else:
            raise ValueError("Unknown %s" % self.dataset)
        # data size
        self.data_size = self.data.shape[0]
        # datashape
        self.data_shape = datashapes[self.dataset]
        # batch size
        self.batch_size = opts["batch_size"]
        # splitting and fill var
        train, test = self._split(opts, self.data, labels, seed)
        self.data_train, self.labels_train = train[0], train[1]
        self.data_test, self.labels_test = test[0], test[1]
        # build tf.dataset
        self.dataset_train, self.iterator_train = self._build_tfdataset(opts, train[0])
        self.dataset_test, self.iterator_test = self._build_tfdataset(opts, test[0])
        # Global iterator
        self.handle = tf.compat.v1.placeholder(tf.string, shape=[])
        self.next_element = tf.compat.v1.data.Iterator.from_string_handle(
            self.handle,
            tf.compat.v1.data.get_output_types(self.dataset_train),
            tf.compat.v1.data.get_output_shapes(self.dataset_train),
        ).get_next()

    def _load_mnist(self, opts, zalando=False, modified=False):
        """Load data from MNIST."""

        self.data_dir = _data_dir(opts)
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None
        with gzip.open(os.path.join(self.data_dir, "train-images-idx3-ubyte.gz")) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000 * 28 * 28 * 1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)
        with gzip.open(os.path.join(self.data_dir, "train-labels-idx1-ubyte.gz")) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(60000), dtype=np.uint8)
            tr_Y = loaded.reshape((60000)).astype(np.int)
        with gzip.open(os.path.join(self.data_dir, "t10k-images-idx3-ubyte.gz")) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000 * 28 * 28 * 1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)
        with gzip.open(os.path.join(self.data_dir, "t10k-labels-idx1-ubyte.gz")) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(10000), dtype=np.uint8)
            te_Y = loaded.reshape((10000)).astype(np.int)
        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)
        X = np.concatenate((tr_X, te_X), axis=0)
        Y = np.concatenate((tr_Y, te_Y), axis=0)

        return X / 255.0, Y

    def _load_smallNORB(self, opts):
        """Load data from smallNORB dataset"""

        # create data_dir and download data if needed
        self.data_dir = _data_dir(opts)
        SMALLNORB_CHUNKS = [
            "smallnorb-5x46789x9x18x6x2x96x96-training-{0}.mat.gz",
            "smallnorb-5x01235x9x18x6x2x96x96-testing-{0}.mat.gz",
        ]
        list_of_images = []
        list_of_labels = []
        for chunk_name in SMALLNORB_CHUNKS:
            # Loading data
            file_path = os.path.join(data_dir, chunk_name.format("dat"))
            with gzip.open(file_path, mode="rb") as f:
                header = _parse_smallNORB_header(f)
                num_examples, channels, height, width = header["dimensions"]
                images = np.zeros(
                    shape=(num_examples, 2, height, width), dtype=np.uint8
                )
                for i in range(num_examples):
                    # Read raw image data and restore shape as appropriate
                    image = struct.unpack(
                        "<" + height * width * "B", f.read(height * width)
                    )
                    image = np.uint8(np.reshape(image, newshape=(height, width)))
                    images[i] = image
            list_of_images.append(_resize_images(images[:, 0]))
            # Loading category
            file_path = os.path.join(data_dir, chunk_name.format("cat"))
            with gzip.open(file_path, mode="rb") as f:
                header = _parse_smallNORB_header(f)
                (num_examples,) = header["dimensions"]
                struct.unpack("<BBBB", f.read(4))  # ignore this integer
                struct.unpack("<BBBB", f.read(4))  # ignore this integer
                categories = np.zeros(shape=num_examples, dtype=np.int32)
                for i in tqdm(
                    range(num_examples), disable=True, desc="Loading categories..."
                ):
                    (category,) = struct.unpack("<i", f.read(4))
                    categories[i] = category
        X = np.concatenate(list_of_images, axis=0)
        X = np.expand_dims(X, axis=-1)
        Y = np.concatenate(categories, axis=0)

        return X, Y

    def _load_celebA(self, opts):
        """Load CelebA"""

        # create data_dir and download data if needed
        self.data_dir = _data_dir(opts)
        # read data
        X = np.array(
            [
                os.path.join(self.data_dir, "img_align_celeba", "%.6d.jpg") % i
                for i in range(1, opts["dataset_size"] + 1)
            ]
        )

        return X, None

    def _split(self, opts, data, labels=None, seed=None):
        """Helper to split data"""

        # splitting data
        if seed is not None:
            np.random.seed(seed)
        idx_random = np.random.permutation(self.data_size)
        np.random.seed()
        if (
            opts["train_dataset_size"] == -1
            or opts["train_dataset_size"] > self.data_size - 10000
        ):
            tr_stop = self.data_size - 10000
        else:
            tr_stop = opts["train_dataset_size"]
        data_train = data[idx_random[:tr_stop]]
        data_test = data[idx_random[-10000:]]
        if labels is not None:
            labels_train = labels[idx_random[:tr_stop]]
            labels_test = labels[idx_random[-10000:]]
        else:
            labels_train, labels_test = None, None
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]

        return (data_train, labels_train), (data_test, labels_test)

    def _build_tfdataset(self, opts, data):
        def _process_path(file_path):
            """Helper to map files paths to image with tf.io.decode_jpeg"""
            # reading .jpg file
            image_file = tf.io.read_file(file_path)
            im_decoded = tf.cast(
                tf.image.decode_jpeg(image_file, channels=3), dtype=tf.dtypes.float32
            )
            # crop and resize
            width = 178
            height = 218
            new_width = 140
            new_height = 140
            if self.crop_style == "closecrop":
                # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
                left = (width - new_width) / 2
                top = (height - new_height) / 2
                right = (width + new_width) / 2
                bottom = (height + new_height) / 2
                im = tf.image.crop_and_resize(
                    tf.expand_dims(im_decoded, axis=0),
                    np.array(
                        [
                            [
                                top / (height - 1),
                                right / (width - 1),
                                bottom / (height - 1),
                                left / (width - 1),
                            ]
                        ]
                    ),
                    [
                        0,
                    ],
                    (64, 64),
                    method="bilinear",
                    extrapolation_value=0,
                )
            else:
                assert False, "{} not implemented.".format(self.crop_style)
            return tf.reshape(im, datashapes["celebA"]) / 255.0

        # Create tf.dataset
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # process path if celeba
        if opts["dataset"] == "celebA":
            dataset = dataset.map(
                _process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        # normalize data if needed
        if opts["input_normalize_sym"]:
            dataset = dataset.map(
                lambda x: (x - 0.5) * 2.0,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        # Shuffle dataset
        dataset = dataset.shuffle(buffer_size=50 * opts["batch_size"])
        # repeat for multiple epochs
        dataset = dataset.repeat()
        # Random batching
        dataset = dataset.batch(batch_size=opts["batch_size"])
        # Prefetch
        dataset = dataset.prefetch(buffer_size=4 * opts["batch_size"])
        # Iterator for each split
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

        return dataset, iterator

    def init_iterator(self, sess):
        sess.run([self.iterator_train.initializer, self.iterator_test.initializer])
        # handle = sess.run(iterator.string_handle())
        train_handle, test_handle = sess.run(
            [self.iterator_train.string_handle(), self.iterator_test.string_handle()]
        )

        return train_handle, test_handle

    def sample_observations(self, keys, dataset="test"):
        if dataset == "test":
            data, labels = self.data_test, self.labels_test
        elif dataset == "train":
            data, labels = self.data_train, self.labels_train
        else:
            raise ValueError("Unknown %s type for sampling" % dataset)
        # data
        if len(data.shape) > 1:
            # all_data is an np.ndarray already loaded into the memory
            if np.amax(data[keys]) > 1:
                x = data[keys] / 255.0
            else:
                x = data[keys]
        else:
            # all_data is a 1d array of paths
            x = []
            for key in list(keys):
                img = self._read_image(key)
                x.append(img)
            x = np.stack(x)
        # labels
        if labels is not None:
            y = labels[keys]
        else:
            y = None

        return x.astype(dtype=np.float32), y

    def _read_image(self, key):
        seed = 123
        assert (
            key == int(os.path.split(self.data[key])[1][:-4]) - 1
        ), "Mismatch between key and img_file_name"
        if self.dataset == "celebA":
            point = self._read_celeba_image(self.data[key])
        else:
            raise Exception(
                "Disc read for {} not implemented yet...".format(self.dataset)
            )

        return point

    def _read_celeba_image(self, file_path):
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(file_path)
        if self.crop_style == "closecrop":
            # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            im = im.crop((left, top, right, bottom))
            im = im.resize((64, 64), Image.ANTIALIAS)
        elif self.crop_style == "resizecrop":
            # This method was used in ALI, AGE, ...
            im = im.resize((64, 78), Image.ANTIALIAS)
            im = im.crop((0, 7, 64, 64 + 7))
        else:
            raise Exception("Unknown crop style specified")
        im_array = np.array(im).reshape(datashapes["celebA"]) / 255.0
        im.close()
        return im_array
