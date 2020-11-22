import tensorflow as tf


def discriminator_reconstruction_loss(real_image, decoded_image):
    return tf.keras.losses.MSE(real_image, decoded_image)


def discriminator_real_fake_loss(real_fake_output_logits_on_real_images, real_fake_output_logits_on_fake_images):
    # label_smoothing_noise = tf.random.normal(tf.shape(real_fake_output_logits_on_real_images))
    # real_fake_output_logits_on_real_images = (label_smoothing_noise * 0.2) + (
    #         0.8 * real_fake_output_logits_on_real_images)
    # real_loss = tf.minimum(0.0, -1 + real_fake_output_logits_on_real_images)
    # real_loss = -1 * tf.reduce_mean(real_loss)

    # label_smoothing_noise = tf.random.normal(tf.shape(real_fake_output_logits_on_fake_images))
    # real_fake_output_logits_on_fake_images = (label_smoothing_noise * 0.2) + (
    #         0.8 * real_fake_output_logits_on_fake_images)
    # fake_loss = tf.minimum(0.0, -1 - real_fake_output_logits_on_fake_images)
    # fake_loss = -1 * tf.reduce_mean(fake_loss)

    loss = tf.reduce_mean(
        tf.nn.relu(1 + real_fake_output_logits_on_real_images) + tf.nn.relu(1 - real_fake_output_logits_on_fake_images))

    return loss


def generator_loss(real_fake_output_logits_on_fake_images):
    # return -1 * tf.reduce_mean(real_fake_output_logits_on_fake_images)
    return tf.reduce_mean(real_fake_output_logits_on_fake_images)
