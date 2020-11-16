import tensorflow as tf


def create_input_noise(batch_size: int):
    return tf.random.normal(shape=(batch_size, 1, 1, 256), mean=0.0, stddev=1.0, dtype=tf.float32)


def read_image_from_path(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return image


def preprocess_images(images):
    """
    Resize and normalize the images tot he range [-1, 1]
    Args:
        images: batch of images (B, H, W, C)

    Returns:
        resized and normalized images
    """

    images = tf.image.resize(images, (1024, 1024))
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


def create_dataset(batch_size: int, folder: str, use_flip_augmentation: bool = True, image_extension: str = "jpg"):
    dataset = tf.data.Dataset.list_files(folder + f"/*.{image_extension}")
    dataset = dataset.map(read_image_from_path)
    if use_flip_augmentation:
        dataset = dataset.map(tf.image.flip_left_right)
    dataset = dataset.map(preprocess_images)
    dataset = dataset.repeat().shuffle(buffer_size=100).batch(batch_size)
    return dataset
