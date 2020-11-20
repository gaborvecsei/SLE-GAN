import unittest

import numpy as np
import tensorflow as tf

from sle_gan.network.discriminator import Discriminator


class TestGeneratorModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.batch_size = 4

    def create_model_and_make_prediction(self, resolution):
        tf.keras.backend.clear_session()
        model = Discriminator(resolution)
        sample_input = np.random.uniform(low=-1, high=1, size=(self.batch_size, resolution, resolution, 3))
        real_fake_logits, image, image_cropped = model(sample_input)
        return real_fake_logits.numpy(), image.numpy(), image_cropped.numpy()

    def test_output_shape(self):
        for resolution in [256, 512, 1024]:
            real_fake_logits, image, image_cropped = self.create_model_and_make_prediction(resolution)
            self.assertEqual(image.shape, (self.batch_size, 128, 128, 3))
            self.assertEqual(real_fake_logits.shape, (self.batch_size, 5, 5, 1))
