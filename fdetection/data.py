import os
import torch

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
from typing import Optional


class DataProcessing:
    """
    Identifies faces in images. Original MTCNN implementation and pipeline available
    in the `facenet_pytorch` library. [1]
    
    References
    ----------
    [1] https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
    """
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
            self.mtcnn(img, save_path=f"output/{output_path}_output.jpg")

        # print message for the next MANUAL step: image "annotation"
        message = """
        All raw files have been processes and face images have been extracted.
        Now we have to separate images manually base on each person's face.

        Navigate to the `./output/` directory and sort images in their respective
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

    # TODO
    def split_data(self):
        """
        Creates different splits for training, validation, and test. This will identify all files
        and their respective classes, then it'll separate randomly those files into respective
        `train`, `val`, and `test` directories.
        """
        pass

    def process(self, path:str) -> torch.Tensor:
        """Processes single image file returning Tensor; useful for inference pipelines."""
        img = Image.open(path)
        return self.mtcnn(img)
