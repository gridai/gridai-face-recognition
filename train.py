import os

from argparse import ArgumentParser
from fdetection import FaceClassifier


if __name__ == '__main__':
    # define CLI arguments
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--backbone', type=str, default="resnet18")
    parser.add_argument('--data_dir', type=str, default=os.getcwd())
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpus', type=int, default=0)
    args = parser.parse_args()

    # start training
    model = FaceClassifier(**args)
    model.train()
