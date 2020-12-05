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
    parser.add_argument("--G-learning-rate", type=float, default=2e-4, help="Learning rate for the Generator")
    parser.add_argument("--D-learning-rate", type=float, default=2e-4, help="Learning rate for the Discriminator")
    parser.add_argument("--diff-augment", action="store_true", help="Apply diff augmentation")
    parser.add_argument("--fid", action="store_true", help="If this is used, FID will be evaluated")
    parser.add_argument("--fid-frequency", type=int, default=1, help="FID will be evaluated at this frequency (epochs)")
    parser.add_argument("--fid-number-of-images", type=int, default=128,
                        help="This many images will be used for the FID calculation")
    args = parser.parse_args()
    return args
