import tensorflow as tf

binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_reconstruction_loss(real_image, decoded_image):
    return tf.keras.losses.MSE(real_image, decoded_image)


def discriminator_real_fake_loss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def discriminator_loss(real_output, fake_output, real_image_128, decoded_image_128):
    real_fake_loss = discriminator_real_fake_loss(real_output=real_output, fake_output=fake_output)
    reconstruction_loss = discriminator_reconstruction_loss(real_image=real_image_128, decoded_image=decoded_image_128)
    return real_fake_loss + reconstruction_loss


def generator_loss(fake_output):
    return binary_crossentropy(tf.ones_like(fake_output), fake_output)
