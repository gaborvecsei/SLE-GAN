import tensorflow as tf


class GLU_BAD(tf.keras.layers.Layer):
    """
    https://arxiv.org/pdf/1612.08083.pdf
    https://leimao.github.io/blog/Gated-Linear-Units/
    """

    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")

    def call(self, inputs, **kwargs):
        x1 = self.conv_1(inputs)
        x1 = tf.nn.sigmoid(x1)
        x2 = self.conv_2(inputs)
        return tf.multiply(x1, x2)


class GLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x_1 = inputs
        x_2 = tf.nn.sigmoid(inputs)
        return x_1 * x_2
