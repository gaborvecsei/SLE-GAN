from functools import partial

import tensorflow as tf


def create_input_noise(batch_size: int):
    return tf.random.normal(shape=(batch_size, 1, 1, 256), mean=0.0, stddev=1.0, dtype=tf.float32)


def read_image_from_path(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return image


def preprocess_images(images, resolution: int):
    """
    Resize and normalize the images tot he range [-1, 1]
    Args:
        images: batch of images (B, H, W, C)

    Returns:
        resized and normalized images
    """

    images = tf.image.resize(images, (resolution, resolution))
    images = tf.cast(images, tf.float32) - 127.5
    images = images / 127.5
    return images


def postprocess_images(images, dtype=tf.float32):
    """
    De-Normalize the images to the range [0, 255]
    Args:
        images: batch of normalized images
        dtype: target dtype

    Returns:
        de-normalized images
    """

    images = (images * 127.5) + 127.5
    images = tf.cast(images, dtype)
    return images


def create_dataset(batch_size: int,
                   folder: str,
                   resolution: int,
                   use_flip_augmentation: bool = True,
                   image_extension: str = "jpg",
                   shuffle_buffer_size: int = 100):
    dataset = tf.data.Dataset.list_files(folder + f"/*.{image_extension}")
    dataset = dataset.map(read_image_from_path)
    if use_flip_augmentation:
        dataset = dataset.map(tf.image.flip_left_right)
    dataset = dataset.map(partial(preprocess_images, resolution=resolution))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def center_crop_images(images, crop_resolution: int):
    """
    Crops the center of the images
    Args:
        images: shape: (B, H, W, 3), H should be equal to W
        crop_resolution: target resolution for the crop

    Returns:
        cropped images which has the shape: (B, crop_resolution, crop_resolution, 3)
    """

    crop_resolution = tf.cast(crop_resolution, tf.float32)
    half_of_crop_resolution = crop_resolution / 2
    image_height = tf.cast(tf.shape(images)[1], tf.float32)
    image_center = image_height / 2

    from_ = int(image_center - half_of_crop_resolution)
    to_ = int(image_center + half_of_crop_resolution)

    return images[:, from_:to_, from_:to_, :]


def get_test_images(batch_size: int, folder: str, resolution: int):
    dataset = create_dataset(batch_size, str(folder), resolution=resolution, use_flip_augmentation=False,
                             shuffle_buffer_size=1)
    for x in dataset.take(1):
        return x
