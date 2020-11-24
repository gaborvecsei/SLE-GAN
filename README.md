# Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis

*Unofficial implementation*, with understandability in mind (verbose implementation)

- [Paper](https://openreview.net/forum?id=1Fqg133qRaI)

The implementation tries to replicate the results from the paper.
What is not discussed in the paper (e.g. filter sizes, training scheduling, hyper parameters), is chosen based on some
experiments and previous knowledge.

## Results

Coming soon

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
- Evaluation (*FID score*) when training
- More advanced training schedule (e.g. learning rate scheduling and initial hyper parameters)
- Random cropping `I_{part}` (right now it is center crop) for the Discriminator
- More exhausting tests for the Generator and Discriminator