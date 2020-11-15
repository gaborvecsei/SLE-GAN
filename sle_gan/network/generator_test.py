import unittest
import generator
import numpy as np


class TestGeneratorModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.batch_size = 4
        self.sample_input = np.random.normal(0, 1, size=(self.batch_size, 1, 1, 256))
        self.model = generator.Generator()

    def make_prediction(self):
        return self.model(self.sample_input).numpy()

    def test_output_shape(self):
        image = self.make_prediction()
        self.assertEqual(image.shape, (self.batch_size, 1024, 1024, 3))

    def test_output_value_range(self):
        image = self.make_prediction()

        min_val = image.min()
        self.assertGreaterEqual(min_val, -1)

        max_val = image.max()
        self.assertLessEqual(max_val, 1)
