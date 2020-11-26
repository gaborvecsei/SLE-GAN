# Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis

*Unofficial implementation*, with understandability in mind (verbose implementation)

> Why the name *SLE-GAN*? Because the paper introduces a new block in the Generator network called *Skip-Layer Excitation (SLE)*

- [Paper](https://openreview.net/forum?id=1Fqg133qRaI)

<img src="art/flower_interpolation_512.png" width="400"></a> <img src="art/generated_flowers_512.png" width="400"></a>

*512x512 generated images (randomly selected) trained for 9 hours with batch size of 8 on [Oxford 17 flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html)
which contains only 1360 images*

The implementation tries to replicate the results from the paper based only on the publication.

What is not discussed in the paper (e.g. filter sizes, training scheduling, hyper parameters), is chosen based on some
experiments and previous knowledge.

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

- Work with various image extensions not just `jpg`
- Add a docker image and requirements
- Name network layers
- Evaluation (*FID score*) when training
- More advanced training schedule (e.g. learning rate scheduling and initial hyper parameters)
- Random cropping `I_{part}` (right now it is center crop) for the Discriminator
- More exhausting tests for the Generator and Discriminator

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
