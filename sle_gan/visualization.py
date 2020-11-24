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


def reconstruct_and_save_images(D, epoch, images, save_folder):
    save_folder = Path(save_folder) / "reconstructions"

    _, x_image_decoded_128, _ = D(images, training=False)
    x_image_decoded_128 = sle_gan.postprocess_images(x_image_decoded_128)
    x_image_decoded_128 = x_image_decoded_128.numpy().astype(np.uint8)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(len(axs)):
        image = x_image_decoded_128[i]
        axs[i].imshow(image)
        axs[i].axis('off')

    fig.set_tight_layout(True)
    save_path = save_folder / f"generated_{str(epoch).zfill(6)}.jpg"
    fig.savefig(str(save_path))
    plt.close()


def generate_and_save_images(G, epoch, test_input, save_folder):
    save_folder = Path(save_folder) / "generated_images"

    predictions = G(test_input, training=False)
    predictions = sle_gan.postprocess_images(predictions)
    predictions = predictions.numpy().astype(np.uint8)

    assert predictions.min() >= 0
    assert predictions.max() <= 255

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(len(axs)):
        image = predictions[i]
        axs[i].imshow(image)
        axs[i].axis('off')

    fig.set_tight_layout(True)
    save_path = save_folder / f"reconstruction_{str(epoch).zfill(6)}.jpg"
    fig.savefig(str(save_path))
    plt.close()
