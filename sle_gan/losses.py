import tensorflow as tf
from .lpips_tf import lpips

def discriminator_reconstruction_loss(real_image, decoded_image):
    # replaced MAE loss with perception loss from official implementation. 

    # expects images in [0,1] 
    distance_percept = lpips((real_image + 1)/2, (decoded_image + 1)/2, model='net-lin', net='vgg')
    return tf.reduce_mean(distance_percept)



def discriminator_real_fake_loss(real_fake_output_logits_on_real_images, real_fake_output_logits_on_fake_images):
    real_loss = tf.minimum(0.0, -1 + real_fake_output_logits_on_real_images)
    real_loss = -1 * tf.reduce_mean(real_loss)

    fake_loss = tf.minimum(0.0, -1 - real_fake_output_logits_on_fake_images)
    fake_loss = -1 * tf.reduce_mean(fake_loss)

    return real_loss + fake_loss


def generator_loss(real_fake_output_logits_on_fake_images):
    return -1 * tf.reduce_mean(real_fake_output_logits_on_fake_images)
