from pathlib import Path
from argparse import ArgumentParser
from fdetection import DataProcessing


if __name__ == '__main__':

    # define CLI arguments
    parser = ArgumentParser()
    parser.add_argument('--raw-data-path', type=str, default=Path().cwd() / Path("raw"))
    args = parser.parse_args()

    # process raw data
    model = DataProcessing(**vars(args))
    model.process_raw_folder()
