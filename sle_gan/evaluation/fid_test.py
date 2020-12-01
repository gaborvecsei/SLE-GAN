import shutil
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sle_gan


def write_images_to_disk(images):
    tmp_folder = Path(tempfile.mkdtemp())
    file_list = []
    for i in range(len(images)):
        file_path = tmp_folder / f"{i}.jpg"
        plt.imsave(file_path, images[i])
        file_list.append(str(file_path))
    return tmp_folder, file_list


class TestFidScore(unittest.TestCase):

    def setUp(self) -> None:
        nb_images = 4
        zero_images = np.zeros((nb_images, 224, 224, 3), dtype=np.uint8)
        one_images = np.ones((nb_images, 224, 224, 3), dtype=np.uint8)

        self.zero_folder, self.zero_files = write_images_to_disk(zero_images)

        self.one_folder, self.one_files = write_images_to_disk(one_images)

    def tearDown(self) -> None:
        shutil.rmtree(self.zero_folder)
        shutil.rmtree(self.one_folder)

    def test_fid_for_zeros_vs_ones(self):
        fid_score = sle_gan.get_fid_score(self.zero_files, self.one_files, 1, 100, 100)
        print(fid_score)
        self.assertGreaterEqual(fid_score, 0)
