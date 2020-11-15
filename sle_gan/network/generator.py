import tensorflow as tf
import tensorflow_addons as tfa


class GLU(tf.keras.layers.Layer):
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


class InputBlock(tf.keras.layers.Layer):
    """
    Input Block

    Input shape: (B, 1, 1, 256)
    Output shape: (B, 4, 4, 256)
    """

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                                kernel_size=(1, 1),
                                                                strides=(4, 4),
                                                                padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.glu = GLU(filters=filters, kernel_size=1)

    def call(self, inputs, **kwargs):
        x = self.conv2d_transpose(inputs)
        x = self.normalization(x)
        x = self.glu(x)
        return x


class UpSamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv2d = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.glu = GLU(filters=self.filters, kernel_size=1)

    def call(self, inputs, **kwargs):
        x = self.upsampling(inputs)
        x = self.conv2d(x)
        x = self.normalization(x)
        x = self.glu(x)
        return x


class SkipLayerExcitationBlock(tf.keras.layers.Layer):
    """
    Skip-Layer Excitation Block

    This block receives 2 feature maps, a high and a low resolution one. Then transforms the low resolution feature map
    and at the end it is multiplied along the channel dimension with the high resolution input.

    E.g.:
    Inputs:
        - High_res shape: (B, 128, 128, 64)
        - Low_res shape: (B, 8, 8, 512)
    Output:
        - shape: (B, 128, 128, 64)
    """

    def __init__(self, input_low_res: UpSamplingBlock, input_high_res: UpSamplingBlock, **kwargs):
        super().__init__(**kwargs)

        self.pooling = tfa.layers.AdaptiveAveragePooling2D(output_size=(4, 4), data_format="channels_last")
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=input_low_res.filters, kernel_size=(4, 4), padding="valid")
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=input_high_res.filters, kernel_size=(1, 1), padding="same")

    def call(self, inputs, **kwargs):
        x_low, x_high = inputs

        x = self.pooling(x_low)
        x = self.conv2d_1(x)
        x = self.leaky_relu(x)
        x = self.conv2d_2(x)
        x = tf.nn.sigmoid(x)

        return x * x_high


class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same")

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = tf.nn.tanh(x)
        return x


class Generator(tf.keras.models.Model):
    """
    Input of the Generator is in shape: (B, 1, 1, 256)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_block = InputBlock(filters=512)  # --> (B, 4, 4, 512)
        self.upsample_8 = UpSamplingBlock(512)  # --> (B, 8, 8, 512)
        self.upsample_16 = UpSamplingBlock(256)  # --> (B, 16, 16, 256)
        self.upsample_32 = UpSamplingBlock(256)  # --> (B, 32, 32, 128)
        self.upsample_64 = UpSamplingBlock(64)  # --> (B, 64, 64, 64)
        self.upsample_128 = UpSamplingBlock(64)  # --> (B, 128, 128, 64)
        self.upsample_256 = UpSamplingBlock(32)  # --> (B, 256, 256, 32)
        self.upsample_512 = UpSamplingBlock(16)  # --> (B, 512, 512, 16)

        self.sle_8_128 = SkipLayerExcitationBlock(self.upsample_8, self.upsample_128)  # --> (B, 128, 128, 64)
        self.sle_16_256 = SkipLayerExcitationBlock(self.upsample_16, self.upsample_256)  # --> (B, 256, 256, 32)
        self.sle_32_512 = SkipLayerExcitationBlock(self.upsample_32, self.upsample_512)  # --> (B, 512, 512, 16)

        self.upsample_1024 = UpSamplingBlock(3)  # --> (B, 1024, 1024, 3)
        self.output_1024 = OutputBlock()  # --> (B, 1024, 1024, 3)

    def call(self, inputs, training=None, mask=None):
        x = self.input_block(inputs)

        x_8 = self.upsample_8(x)
        x_16 = self.upsample_16(x_8)
        x_32 = self.upsample_32(x_16)
        x_64 = self.upsample_64(x_32)

        x_128 = self.upsample_128(x_64)
        x_sle_128 = self.sle_8_128([x_8, x_128])

        x_256 = self.upsample_256(x_sle_128)
        x_sle_256 = self.sle_16_256([x_16, x_256])

        x_512 = self.upsample_512(x_sle_256)
        x_sle_512 = self.sle_32_512([x_32, x_512])

        x_1024 = self.upsample_1024(x_sle_512)
        image = self.output_1024(x_1024)

        return image
