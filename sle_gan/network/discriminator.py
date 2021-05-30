import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import SpectralNormalization as SN
from sle_gan import center_crop_images
from sle_gan.network.common_layers import GLU


class InputBlock(tf.keras.layers.Layer):
    def __init__(self, downsampling_factor: int, filters, **kwargs):
        super().__init__(**kwargs)
        assert downsampling_factor in [1, 2, 4]

        conv_1_strides = 2
        conv_2_strides = 2

        if downsampling_factor <= 2:
            conv_2_strides = 1

        if downsampling_factor == 1:
            conv_1_strides = 1

        self.conv_1 = SN(tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=conv_1_strides, padding="same", use_bias=False))
        self.activation_1 = tf.keras.layers.LeakyReLU(0.2)
        self.conv_2 = SN(tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=conv_2_strides, padding="same", use_bias=False))
        self.normalization = tf.keras.layers.LayerNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(0.2)

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

        self.conv_1 = SN(tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding="same", use_bias=False))
        self.normalization_1 = tf.keras.layers.LayerNormalization()
        self.activation_1 = tf.keras.layers.LeakyReLU(0.2)

        self.conv_2 = SN(tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False))
        self.normalization_2 = tf.keras.layers.LayerNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(0.2)
    
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
        self.conv = SN(tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding="valid", use_bias=False))
        self.normalization = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.2)

    def call(self, inputs, **kwargs):
        x = self.pooling(inputs)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class DownSamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.down_1 = DownSamplingBlock1(filters)
        self.down_2 = DownSamplingBlock2(filters)

    def call(self, inputs, **kwargs):
        x_1 = self.down_1(inputs)
        x_2 = self.down_2(inputs)
        return (x_1 + x_2)/2 


class SimpleDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, output_filters, **kwargs):
        super().__init__(**kwargs)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv = SN(tf.keras.layers.Conv2D(filters=output_filters * 2, kernel_size=3, padding="same", use_bias=False))
        self.normalization = tf.keras.layers.LayerNormalization()
        self.glu = GLU()

    def call(self, inputs, **kwargs):
        x = self.upsampling(inputs)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.glu(x)
        return x


class SimpleDecoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder_block_filter_sizes = [256, 128, 128, 64]
        self.decoder_blocks = [SimpleDecoderBlock(output_filters=x) for x in self.decoder_block_filter_sizes]
        self.conv_output = SN(tf.keras.layers.Conv2D(3, 1, 1, padding="same", use_bias=False))

    def call(self, inputs, **kwargs):
        x = inputs
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        x = self.conv_output(x)
        x = tf.nn.tanh(x)
        return x


class RealFakeOutputBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = SN(tf.keras.layers.Conv2D(filters=filters, kernel_size=1))
        self.normalization = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.2)
        self.conv_2 = SN(tf.keras.layers.Conv2D(filters=1, kernel_size=4, use_bias=False))

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.conv_2(x)
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

    def __init__(self, input_low_res_filters: int, input_high_res_filters: int, **kwargs):
        super().__init__(**kwargs)

        self.pooling = tfa.layers.AdaptiveAveragePooling2D(output_size=(4, 4), data_format="channels_last")
        self.conv2d_1 = SN(tf.keras.layers.Conv2D(filters=input_low_res_filters,
                                               kernel_size=(4, 4),
                                               strides=1,
                                               padding="valid", use_bias=False))
        #replace with silu (aka swift)
        self.conv2d_2 = SN(tf.keras.layers.Conv2D(filters=input_high_res_filters,
                                               kernel_size=(1, 1),
                                               strides=1,
                                               padding="valid", use_bias=False))

    def call(self, inputs, **kwargs):
        x_low, x_high = inputs

        x = self.pooling(x_low)
        x = self.conv2d_1(x)
        x = tf.nn.silu(x)
        x = self.conv2d_2(x)
        x = tf.nn.sigmoid(x)

        return x * x_high

class Discriminator(tf.keras.models.Model):
    def __init__(self, input_resolution: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert input_resolution in [256, 512, 1024], "Resolution should be 256 or 512 or 1024"
        self.input_resolution = input_resolution

        downsampling_factor_dict = {256: 1, 512: 2, 1024: 4}
        input_block_filters_dict = {256: 8, 512: 16, 1024: 32}
        self.input_block = InputBlock(filters=input_block_filters_dict[input_resolution],
                                      downsampling_factor=downsampling_factor_dict[input_resolution])

        self.downsample_128 = DownSamplingBlock(filters=64) #should be 32
        self.downsample_64 = DownSamplingBlock(filters=128) #should be 64
        self.downsample_32 = DownSamplingBlock(filters=128) #should be 128
        self.downsample_16 = DownSamplingBlock(filters=256) #should be 256
        self.downsample_8 = DownSamplingBlock(filters=512)  #should be 512

        # added sle blocks into discriminator per official implementation.
        self.sle_in_32 = SkipLayerExcitationBlock(256, self.downsample_32.filters)
        self.sle_128_16 = SkipLayerExcitationBlock(self.downsample_128.filters, self.downsample_16.filters)
        self.sle_64_8 = SkipLayerExcitationBlock(self.downsample_64.filters, self.downsample_8.filters)
    
        #TODO: self.decoder_big 
        self.decoder_image_part = SimpleDecoder()
        self.decoder_image = SimpleDecoder()

        self.real_fake_output = RealFakeOutputBlock(filters=256)

    def initialize(self, batch_size: int = 1):
        sample_input = tf.random.uniform(shape=(batch_size, self.input_resolution, self.input_resolution, 3), minval=-1,
                                         maxval=1, dtype=tf.float32)
        sample_output = self.call(sample_input)
        return sample_output

    @tf.function
    def call(self, inputs, training=None, mask=None):
        
        x_in = self.input_block(inputs)  # --> (B, 256, 256, F)
        x_128 = self.downsample_128(x_in)  # --> (B, 128, 128, 64)
        x_64 = self.downsample_64(x_128)  # --> (B, 64, 64, 128)
        
        x_32 = self.downsample_32(x_64)  # --> (B, 32, 32, 128)
        x_sle_32 = self.sle_in_32([x_in, x_32])
        
        x_16 = self.downsample_16(x_sle_32)  # --> (B, 16, 16, 256)
        x_sle_16 = self.sle_128_16([x_128, x_16])
        
        x_8 = self.downsample_8(x_sle_16)  # --> (B, 8, 8, 512)
        x_sle_8 = self.sle_64_8([x_64, x_8])
        
        # Implemented random cropping but left name.
        # DONE: instead of just center cropping implement random cropping
        center_cropped_x_16 = center_crop_images(x_sle_16, 8)  # --> (B, 8, 8, 64)
        x_image_decoded_128_center_part = self.decoder_image_part(center_cropped_x_16)  # --> (B, 128, 128, 3)
        x_image_decoded_128 = self.decoder_image(x_sle_8)  # --> (B, 128, 128, 3)

        x_real_fake_logits = self.real_fake_output(x_sle_8)  # --> (B, 5, 5, 1)

        return x_real_fake_logits, x_image_decoded_128, x_image_decoded_128_center_part
