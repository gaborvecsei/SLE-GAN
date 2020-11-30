from pathlib import Path

import numpy as np
import tensorflow as tf


def read_images(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    # TODO: Remove this
    image = image[..., 1:]
    # TODO: should we resize?
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image


def create_tmp_dataset(file_paths: list, batch_size: int):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(read_images).batch(batch_size=batch_size)
    return dataset


def get_encodings(model: tf.keras.models.Model, dataset: tf.data.Dataset, max_nb_images: int = 10000):
    image_encodings_2048 = np.zeros((max_nb_images, 2048))
    for image_batch in dataset:
        encodings = model(image_batch)
        # TODO: finish this
        image_encodings_2048[0:1] = encodings
    return image_encodings_2048


def get_encoding_statistics(encodings):
    mu = np.mean(encodings, axis=0)
    sigma = np.cov(encodings, rowvar=False)
    return mu, sigma


def calculate_fid_score(real_mu, real_sigma, fake_mu, fake_sigma):
    return 0


def create_inception_model():
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                           weights="imagenet",
                                                           input_shape=(None, None, 3),
                                                           pooling="avg")
    return model


def get_fid_score(real_paths: list, fake_paths: list, batch_size: int = 1):
    # Just to make sure we have the same number of images from both category
    nb_of_images = min(len(real_paths), len(fake_paths))
    real_paths = real_paths[:nb_of_images]
    fake_paths = fake_paths[:nb_of_images]

    model = create_inception_model()

    read_dataset = create_tmp_dataset(real_paths, batch_size)
    real_encodings = get_encodings(model, read_dataset, nb_of_images)
    real_mu, real_sigma = get_encoding_statistics(real_encodings)

    fake_dataset = create_tmp_dataset(fake_paths, batch_size)
    fake_encodings = get_encodings(model, fake_dataset, nb_of_images)
    fake_mu, fake_sigma = get_encoding_statistics(fake_encodings)

    fid_score = calculate_fid_score(real_mu, real_sigma, fake_mu, fake_sigma)

    return fid_score


real_paths = list(Path("../../asdasd/real").glob("*.png"))
real_paths = list(map(str, real_paths))

fake_paths = list(Path("../../asdasd/fake").glob("*.png"))
fake_paths = list(map(str, fake_paths))
fid = get_fid_score(real_paths, fake_paths)
print(fid)
