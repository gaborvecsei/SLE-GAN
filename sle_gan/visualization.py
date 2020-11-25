from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sle_gan


def visualize_and_save_images(epoch: int, images: np.ndarray, save_folder: Path, rows: int, cols: int,
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

