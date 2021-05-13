import os
import torch
import shutil
import numpy as np

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
from typing import Optional, Tuple, List




class DataProcessing:
    """
    Identifies faces in images. Original MTCNN implementation and pipeline available
    in the `facenet_pytorch` library. [1]
    
    References
    ----------
    [1] https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
    """
    # output directory for processed files
    processed_directory = "processed"

    def __init__(self, raw_data_path: Optional[str] = None):
        
        # check that raw data path exits
        if raw_data_path:
            self.raw_data_path = Path(raw_data_path)
            if not self.raw_data_path.exists():
                raise ValueError(f"Raw data path does not exist: {raw_data_path}")

        # creates image cropping pipeling; use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # default optiosn lifted from https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device,
            keep_all=True
        )

    def process_raw_folder(self) -> None:
        """Process images from a folder sequentially"""
        # file all images with expected extension
        for image in tqdm(self.raw_data_path.glob("*.jpg")):

            # generate cropped image pre-pending original file name
            img = Image.open(image)
            output_path = os.path.basename(os.path.splitext(image)[0])
            self.mtcnn(img, save_path=f"{self.processed_directory}/{output_path}_output.jpg")

        # print message for the next MANUAL step: image "annotation"
        message = f"""
        All raw files have been processes and face images have been extracted.
        Now we have to separate images manually base on each person's face.

        Navigate to the `./{self.processed_directory}` directory and sort images in their respective
        classes, i.e. one folder per person. For example:

            dataset/vera/photo_1.jpg
            dataset/vera/photo_2.jpg
            dataset/luis/photo_1.jpg
            ...
        
        That way we'll be able to easily load the images and infer their respective
        classes on training.

        Also, the MTCNN algorithm makes mistakes. Make sure to exclude all images that
        do not contain faces from the results.
        """
        print(message)

    @staticmethod
    def _calculate_n_files(N:int, split:Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[int, int, int]:
        """train, test, validation"""
        return (int(split[0] * N), int(split[1] * N), int(split[2] * N))
    
    @staticmethod
    def _generates_idx_arrays(N:int, samples:Tuple[int, int, int]=(10, 10, 10)) -> List[List[int]]:
        """Generates random arrays of indices for creating different datasets with."""
        idx = list(range(N))

        # generates sample indexes from original dataset
        idxs = []
        for sample in samples:
            selected_samples = np.random.choice(idx, sample, replace=False)
            idxs.append(selected_samples)

            # remove index from array so it isn't sampled any longer
            for i in selected_samples:
                idx.remove(i)

        # expected: train, test, val
        return idxs

    @staticmethod
    def _create_base_directories() -> None:
        """Creates base directories. We first remove the directories to prevent storing duplicate data by accident."""
        directories = [Path("dataset/train"), Path("dataset/test"), Path("dataset/val")]
        for dir in directories:

            # deletes if exists
            if dir.exists():
                shutil.rmtree(dir)
            
            # create directory
            dir.mkdir(parents=True, exist_ok=True)

    def create_training_dataset(self) -> None:
        """
        Creates different splits for training, validation, and test. This will identify all files
        and their respective classes, then it'll separate randomly those files into respective
        `train`, `val`, and `test` directories.
        """
        # finds all manually "annotated" files
        path = Path(self.processed_directory)
        all_files = [f for f in path.glob("**/*.jpg")]

        # creates different random splits
        N = len(all_files)
        train, test, val = self._generates_idx_arrays(N, self._calculate_n_files(N))

        # creates base directories; also cleans data if there's any data available
        self._create_base_directories()

        # copies files into directories
        directories = [Path("dataset/train"), Path("dataset/test"), Path("dataset/val")]
        for dataset_idx, dir in zip((train, test, val), directories):
            for i in dataset_idx:
                parent_dir = dir / all_files[i].parent.name
                parent_dir.mkdir(parents=True, exist_ok = True)
                shutil.copy(all_files[i], parent_dir / Path(all_files[i].name))

        print("Dataset created successfully")

    def process(self, path:str) -> torch.Tensor:
        """Processes single image file returning Tensor; useful for inference pipelines."""
        img = Image.open(path)
        return self.mtcnn(img)
