from pathlib import Path
from argparse import ArgumentParser
from fdetection import FaceClassifier


if __name__ == '__main__':

    # define CLI arguments
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--image_path', type=str)
    args = parser.parse_args()

    # start training
    model_kwargs = vars(args)
    image_path = model_kwargs.pop("image_path")
    model = FaceClassifier(**model_kwargs, inference=True)
    model.predict(image_path)
