import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        channels = tf.shape(inputs)[-1]
        nb_split_channels = channels // 2

        x_1 = inputs[:, :, :, :nb_split_channels]
        x_2 = inputs[:, :, :, nb_split_channels:]

        return x_1 * tf.nn.sigmoid(x_2)
