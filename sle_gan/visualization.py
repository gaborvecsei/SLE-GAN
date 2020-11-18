from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_and_save_images(G, epoch, test_input, save_folder: str):
    save_folder = Path(save_folder) / "generated_images"
    if save_folder.is_dir():
        save_folder.mkdir(parents=True)

    predictions = G(test_input, training=False).numpy()

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs = axs.flatten()

    for i in range(predictions.shape[0]):
        image = predictions[i] * 127.5 + 127.5
        image = image.astype(np.uint8)
        axs[i].imshow(image * 127.5 + 127.5)
        axs[i].axis('off')

    fig.set_tight_layout(True)
    save_path = save_folder / f"image_epoch_{epoch}.jpg"
    fig.savefig(str(save_path))
    plt.close()
