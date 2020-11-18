import tensorflow as tf


def discriminator_reconstruction_loss(real_image, decoded_image):
    return tf.keras.losses.MSE(real_image, decoded_image)


def discriminator_real_fake_loss(real_fake_output_logits_on_real_images, real_fake_output_logits_on_fake_images):
    real_loss = tf.minimum(0, -1 + real_fake_output_logits_on_real_images)
    real_loss = tf.reduce_mean(real_loss)

    fake_loss = tf.minimum(0, -1 - real_fake_output_logits_on_fake_images)
    fake_loss = tf.reduce_mean(fake_loss)
    return real_loss + fake_loss


def discriminator_loss(real_fake_output_logits_on_real_images,
                       real_fake_output_logits_on_fake_images,
                       real_image_128,
                       decoded_image_128):
    real_fake_loss = discriminator_real_fake_loss(
        real_fake_output_logits_on_real_images=real_fake_output_logits_on_real_images,
        real_fake_output_logits_on_fake_images=real_fake_output_logits_on_fake_images)
    reconstruction_loss = discriminator_reconstruction_loss(real_image=real_image_128, decoded_image=decoded_image_128)
    return real_fake_loss + reconstruction_loss


def generator_loss(real_fake_output_logits_on_fake_images):
    return -1 * tf.reduce_mean(real_fake_output_logits_on_fake_images)
