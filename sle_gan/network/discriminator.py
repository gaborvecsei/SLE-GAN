import tensorflow as tf
from sle_gan.network.common_layers import GLU


class InputBlock(tf.keras.layers.Layer):
    """
    Input Block

    Input shape: (B, 1024, 1024, 3)
    Output shape: (B, 256, 256, C)
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding="same")
        self.activation_1 = tf.keras.layers.LeakyReLU(0.1)
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.normalization(x)
        x = self.activation_2(x)
        return x


class DownSamplingBlock1(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding="same")
        self.normalization_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.LeakyReLU(0.1)

        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")
        self.normalization_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.normalization_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.activation_2(x)
        return x


class DownSamplingBlock2(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding="valid")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.pooling(inputs)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class DownSamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        self.down_1 = DownSamplingBlock1(filters)
        self.down_2 = DownSamplingBlock2(filters)

    def call(self, inputs, **kwargs):
        x_1 = self.down_1(inputs)
        x_2 = self.down_2(inputs)
        return x_1 + x_2


class SimpleDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.glu = GLU(filters=filters, kernel_size=1)

    def call(self, inputs, **kwargs):
        x = self.upsampling(inputs)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.glu(x)
        return x


class SimpleDecoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder_block_filter_sizes = [32, 16, 8, 3]
        self.decoder_blocks = [SimpleDecoderBlock(filters=x) for x in self.decoder_block_filter_sizes]

    def call(self, inputs, **kwargs):
        x = inputs
        for decoder in self.decoder_blocks:
            x = decoder(x)
        return x


class RealFakeOutputBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=1)
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.1)
        self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=4)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.conv_2(x)
        return x


class Discriminator(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_block = InputBlock(filters=16)  # --> (B, 256, 256, 16)

        self.downsample_128 = DownSamplingBlock(filters=32)  # --> (B, 128, 128, 32)
        self.downsample_64 = DownSamplingBlock(filters=64)  # --> (B, 64, 64, 64)
        self.downsample_32 = DownSamplingBlock(filters=128)  # --> (B, 32, 32, 128)
        self.downsample_16 = DownSamplingBlock(filters=256)  # --> (B, 16, 16, 256)
        self.downsample_8 = DownSamplingBlock(filters=512)  # --> (B, 8, 8, 512)

        # TODO: I_{part} decoder is not part of this implementation
        # self.decoder_image_part = SimpleDecoder()
        self.decoder_image = SimpleDecoder()  # --> (B, 128, 128, 3)

        self.real_fake_logits = RealFakeOutputBlock()  # --> (B, 5, 5, 1)

    def call(self, inputs, training=None, mask=None):
        x = self.input_block(inputs)

        x = self.downsample_128(x)
        x = self.downsample_64(x)
        x = self.downsample_32(x)
        x = self.downsample_16(x)
        x_8 = self.downsample_8(x)

        # TODO: I_{part} decoder is not part of this implementation

        x_image = self.decoder_image(x_8)
        x_real_fake_logits = self.real_fake_logits(x_8)

        return x_real_fake_logits, x_image
