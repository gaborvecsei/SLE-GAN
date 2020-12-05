from functools import partial

import numpy as np
import tensorflow as tf
from scipy import linalg


@tf.function
def read_images(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image


def create_tmp_dataset(file_paths: list, batch_size: int, image_height: int = None, image_width: int = None):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(read_images)
    if image_width is not None and image_height is not None:
        dataset = dataset.map(partial(tf.image.resize, size=(image_height, image_width)))
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def get_encodings(model: tf.keras.models.Model, dataset: tf.data.Dataset, max_nb_images: int):
    image_encodings_2048 = np.zeros((max_nb_images, 2048))

    for i, image_batch in enumerate(dataset):
        batch_size = np.shape(image_batch)[0]
        encodings = model(image_batch)

        start_index = i * batch_size
        end_index = start_index + batch_size
        image_encodings_2048[start_index:end_index] = encodings

    return image_encodings_2048


def get_encoding_statistics(encodings):
    mu = np.mean(encodings, axis=0)
    sigma = np.cov(encodings, rowvar=False)
    return mu, sigma


def calculate_fid_score_from_mu_and_sigma(real_mu, real_sigma, fake_mu, fake_sigma):
    ssdiff = np.sum((real_mu - fake_mu) ** 2.0)
    covmean = linalg.sqrtm(real_sigma.dot(fake_sigma))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = ssdiff + np.trace(real_sigma) + np.trace(fake_sigma) - 2 * np.trace(covmean)
    return fid_score


class InceptionModel(tf.keras.models.Model):
    def __init__(self, height: int = None, width: int = None):
        super().__init__()
        self.model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                    weights="imagenet",
                                                                    input_shape=(height, width, 3),
                                                                    pooling="avg")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=False)


def calculate_FID(inception_model,
                  real_paths: list,
                  fake_paths: list,
                  batch_size: int = 1,
                  image_height: int = None,
                  image_width: int = None):
    # Just to make sure we have the same number of images from both category
    nb_of_images = min(len(real_paths), len(fake_paths))
    real_paths = real_paths[:nb_of_images]
    fake_paths = fake_paths[:nb_of_images]

    read_dataset = create_tmp_dataset(real_paths, batch_size, image_height, image_width)
    real_encodings = get_encodings(inception_model, read_dataset, nb_of_images)
    real_mu, real_sigma = get_encoding_statistics(real_encodings)

    fake_dataset = create_tmp_dataset(fake_paths, batch_size, image_height, image_width)
    fake_encodings = get_encodings(inception_model, fake_dataset, nb_of_images)
    fake_mu, fake_sigma = get_encoding_statistics(fake_encodings)

    fid_score = calculate_fid_score_from_mu_and_sigma(real_mu, real_sigma, fake_mu, fake_sigma)

    return fid_score
