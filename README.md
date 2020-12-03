# Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis

*Unofficial implementation*, with understandability in mind (verbose implementation)

> Why the name *SLE-GAN*? Because the paper introduces a new block in the Generator network called *Skip-Layer Excitation (SLE)*

- [Paper](https://openreview.net/forum?id=1Fqg133qRaI)

<img src="art/flower_interpolation_512.png" width="400"></a> <img src="art/generated_flowers_512.png" width="400"></a>

> *512x512 generated images (randomly selected) trained for 9 hours with batch size of 8 on [Oxford 17 flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html)
which contains only 1360 images*

The implementation tries to replicate the results from the paper based only on the publication.

What is not discussed in the paper (e.g. filter sizes, training scheduling, hyper parameters), is chosen based on some
experiments and previous knowledge.

## Usage

You can easily use the separate parts of the code. The `Generator` and `Discriminator` are Tensorflow Keras models (`tf.keras.models.Model`)

For example if you'd like to generate new images:

```python
import sle_gan

G = sle_gan.Generator(output_resolution=512)
G.load_weights("generator_weights.h5")

input_noise = sle_gan.create_input_noise(batch_size=1)
generated_images = G(input_noise)
generated_images = sle_gan.postprocess_images(generated_images, tf.uint8).numpy()
```

## Train

```
$ python train.py --help

usage: train.py [-h] [--name NAME] [--override] --data-folder DATA_FOLDER
                [--resolution RESOLUTION]
                [--generator-weights GENERATOR_WEIGHTS]
                [--discriminator-weights DISCRIMINATOR_WEIGHTS]
                [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                [--learning-rate LEARNING_RATE] [--diff-augment]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of the experiment
  --override            Removes previous experiment with same name
  --data-folder DATA_FOLDER
                        Folder with the images
  --resolution RESOLUTION
                        Either 256, 512 or 1024. Default is 512.
  --generator-weights GENERATOR_WEIGHTS
  --discriminator-weights DISCRIMINATOR_WEIGHTS
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --learning-rate LEARNING_RATE
                        Learning rate for both G and D
  --diff-augment        Apply diff augmentation
```

## Todos

- Add a docker image and requirements
- Evaluation (*FID score*) when training
- More advanced training schedule (e.g. learning rate scheduling and initial hyper parameters)
- Random cropping `I_{part}` (right now it is center crop) for the Discriminator

## Citation

```bibtex
@inproceedings{
    anonymous2021towards,
    title={Towards Faster and Stabilized {\{}GAN{\}} Training for High-fidelity Few-shot Image Synthesis},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=1Fqg133qRaI},
    note={under review}
}
```
