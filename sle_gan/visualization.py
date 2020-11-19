from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sle_gan


def generate_and_save_images(G, epoch, test_input, save_folder: str):
    save_folder = Path(save_folder) / "generated_images"
    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)

    predictions = G(test_input, training=False)
    predictions = sle_gan.postprocess_images(predictions)
    predictions = predictions.numpy().astype(np.uint8)

    assert predictions.min() >= 0
    assert predictions.max() <= 255

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs = axs.flatten()

    for i in range(len(axs)):
        image = predictions[i]
        axs[i].imshow(image)
        axs[i].axis('off')

    fig.set_tight_layout(True)
    save_path = save_folder / f"image_epoch_{epoch}.jpg"
    fig.savefig(str(save_path))
    plt.close()
