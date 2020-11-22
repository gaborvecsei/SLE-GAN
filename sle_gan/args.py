import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="experiment", help="Name of the experiment")
    parser.add_argument("--v", type=int, default=1, help="REMOVE ME")
    parser.add_argument("--override", action="store_true", help="Removes previous experiment with same name")
    parser.add_argument("--data-folder", type=str, required=True,
                        help="Folder with the .jpg images (other extension won't work at the moment)")
    parser.add_argument("--resolution", type=int, default=256, help="Either 256, 512 or 1024. Default is 256.")
    parser.add_argument("--generator-weights", type=str, default=None)
    parser.add_argument("--discriminator-weights", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--diff-augment", action="store_true", help="Apply diff augmentation")
    args = parser.parse_args()
    return args
