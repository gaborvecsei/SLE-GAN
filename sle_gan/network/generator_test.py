import unittest

import numpy as np
import tensorflow as tf

from sle_gan.network.generator import Generator


class TestGeneratorModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.batch_size = 4
        self.sample_input = np.random.normal(0, 1, size=(self.batch_size, 1, 1, 256))

    def create_model_and_make_prediction(self, resolution):
        tf.keras.backend.clear_session()
        model = Generator(output_resolution=resolution)
        pred = model(self.sample_input).numpy()
        return pred

    def test_output_shapes(self):
        for resolution in [256, 512, 1024]:
            image = self.create_model_and_make_prediction(resolution)
            self.assertEqual(image.shape, (self.batch_size, resolution, resolution, 3))

    def test_output_value_range(self):
        image = self.create_model_and_make_prediction(256)

        min_val = image.min()
        self.assertGreaterEqual(min_val, -1)

        max_val = image.max()
        self.assertLessEqual(max_val, 1)
