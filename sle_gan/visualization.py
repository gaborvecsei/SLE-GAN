from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sle_gan


def reconstructions(D, epoch, images, save_folder):
    save_folder = Path(save_folder) / "reconstructions"
    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)

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
    save_path = save_folder / f"image_epoch_{epoch}.jpg"
    fig.savefig(str(save_path))
    plt.close()


def generate_and_save_images(G, epoch, test_input, save_folder):
    save_folder = Path(save_folder) / "generated_images"
    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)

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
    save_path = save_folder / f"image_epoch_{epoch}.jpg"
    fig.savefig(str(save_path))
    plt.close()
