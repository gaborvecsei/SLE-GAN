import tensorflow as tf

import sle_gan


@tf.function
def train_step(G, D, G_optimizer, D_optimizer, images) -> tuple:
    batch_size = tf.shape(images)[0]
    noise_input = sle_gan.create_input_noise(batch_size)
    image_batch_128 = tf.image.resize(images, (128, 128))

    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
        generated_images = G(noise_input, training=True)

        real_output, decoded_real_image = D(images, training=True)
        fake_output, _ = D(generated_images, training=True)

        G_loss = sle_gan.generator_loss(fake_output)
        D_loss = sle_gan.discriminator_loss(real_output=real_output,
                                            fake_output=fake_output,
                                            real_image_128=image_batch_128,
                                            decoded_image_128=decoded_real_image)

    G_gradients = tape_G.gradient(G_loss, G.trainable_variables)
    D_gradients = tape_D.gradient(D_loss, D.trainable_variables)

    G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))
    D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables))

    return G_loss, D_loss
