import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_images_on_grid_and_save(epoch: int, images: np.ndarray, save_folder: Path, rows: int, cols: int,
                                      figsize=(10, 10), name_suffix: str = ""):
    assert len(images) == (rows * cols)

    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()

    for i in range(len(axs)):
        image = images[i]
        axs[i].imshow(image)
        axs[i].axis('off')

    fig.set_tight_layout(True)
    save_path = save_folder / f"{str(epoch).zfill(6)}{name_suffix}.jpg"
    fig.savefig(str(save_path))
    plt.close()


def write_images_to_disk(images: np.ndarray, folder: str = None) -> tuple:
    if folder is None:
        folder = Path(tempfile.mkdtemp())

    file_list = []
    for i in range(len(images)):
        file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=str(folder)).name
        plt.imsave(file_path, images[i])
        file_list.append(file_path)

    return folder, file_list
