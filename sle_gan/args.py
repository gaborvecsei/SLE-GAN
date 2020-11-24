import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="experiment", help="Name of the experiment")
    parser.add_argument("--override", action="store_true", help="Removes previous experiment with same name")
    parser.add_argument("--data-folder", type=str, required=True,
                        help="Folder with the images")
    parser.add_argument("--resolution", type=int, default=512, help="Either 256, 512 or 1024. Default is 512.")
    parser.add_argument("--generator-weights", type=str, default=None)
    parser.add_argument("--discriminator-weights", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate for both G and D")
    parser.add_argument("--diff-augment", action="store_true", help="Apply diff augmentation")
    args = parser.parse_args()
    return args
