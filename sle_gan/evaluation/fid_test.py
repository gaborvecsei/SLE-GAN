import shutil
import unittest

import numpy as np

import sle_gan


class TestFidScore(unittest.TestCase):

    def setUp(self) -> None:
        nb_images = 4
        zero_images = np.zeros((nb_images, 224, 224, 3), dtype=np.uint8)
        one_images = np.ones((nb_images, 224, 224, 3), dtype=np.uint8)

        self.zero_folder, self.zero_files = sle_gan.write_images_to_disk(zero_images)
        self.one_folder, self.one_files = sle_gan.write_images_to_disk(one_images)

    def tearDown(self) -> None:
        shutil.rmtree(self.zero_folder)
        shutil.rmtree(self.one_folder)

    def test_fid_for_zeros_vs_ones(self):
        fid_score = sle_gan.calculate_FID(self.zero_files, self.one_files, 1, 100, 100)
        print(fid_score)
        self.assertGreaterEqual(fid_score, 0)
