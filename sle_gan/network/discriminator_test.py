import unittest
from sle_gan.network.discriminator import Discriminator
import numpy as np


class TestGeneratorModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.batch_size = 4
        self.sample_input = np.random.uniform(low=-1, high=1, size=(self.batch_size, 1024, 1024, 3))
        self.model = Discriminator()

    def make_prediction(self):
        real_fake_logits, image = self.model(self.sample_input)
        return real_fake_logits.numpy(), image.numpy()

    def test_output_shape(self):
        real_fake_logits, image = self.make_prediction()
        self.assertEqual(image.shape, (self.batch_size, 128, 128, 3))
        self.assertEqual(real_fake_logits.shape, (self.batch_size, 5, 5, 1))
