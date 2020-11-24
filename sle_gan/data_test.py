import unittest
import sle_gan
import numpy as np


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 4
        self.dataset = sle_gan.create_dataset(self.batch_size, "../dataset", resolution=1024)

        self.image_batch = None
        for x in self.dataset.take(1):
            self.image_batch = x
        self.image_batch = self.image_batch.numpy()

    def test_output_shape(self):
        image_batch_shape = self.image_batch.shape
        self.assertEqual(image_batch_shape, (self.batch_size, 1024, 1024, 3))

    def test_dtype(self):
        self.assertEqual(self.image_batch.dtype, np.float32)

    def test_min_max_values(self):
        self.assertGreaterEqual(self.image_batch.min(), -1)
        self.assertLessEqual(self.image_batch.max(), 1)
