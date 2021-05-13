import os
import timm
import torch

from pathlib import Path
from flash.vision import ImageClassifier
from flash import Trainer
import pytorch_lightning as pl

from fdetection.data import DataProcessing
from flash.vision import ImageClassificationData, ImageClassifier


class FaceClassifier:
    """
    Face classifier detects faces using different backbones. Data must be organized
    assuming the MTCNN pipeline [1] (i.e. faces are detected and cropped into standard
    images).

    References
    ----------
    [1] Joint Face Detection and Alignment usinn Multi-task Cascaded
        Convolutional Networks, Zhang et al., 2016. Available at: https://arxiv.org/pdf/1604.02878.pdf
    """
    def __init__(self, data_path:str, backbone: str = "resnet18", batch_size: int = 32, num_workers: int = 4,
                 seed:int = 1234, learning_rate:float = 0.001, gpus:int = 0, inference: bool = False):
        self.seed = seed
        self.learning_rate = learning_rate

        # if user passes `gpus` parameter use it; otherwise, use the
        # `_PL_TRAINER_GPUS` environment variable automatically populated by
        # Grid indicating how many GPUs are available
        if gpus > 0:
            self.gpus = gpus
        else:
            self.gpus = os.getenv("_PL_TRAINER_GPUS", 0)

        # seed all random number generators
        pl.seed_everything(self.seed)

        # if inference, instantiate data pipeline
        if inference:
            self.data_processing = DataProcessing()
    
    @property
    def model(self) -> ImageClassifier:
        """
        Model representation. This will either use a model from the standard Flash library of implementations
        or use a model from elsewhere (e.g. `timm`).
        """
        # make available other PyTorch models; adds support for timm implementations
        if self.backbone == "resnet200d":
            model = timm.create_model(self.backbone, pretrained=True)
            backbone = (model, model.num_features)
        else:
            backbone = self.backbone

        # create Flash classifier
        model = ImageClassifier(backbone=backbone,
                                num_classes=self.data.num_classes,
                                optimizer=torch.optm.Adam,
                                learning_rate=self.learning_rate)

        return model

    @property
    def data(self) -> ImageClassificationData:
        """Creates DataLoaders if not available"""
        if not hasattr(self, "_data"):
            # define folder location
            data_kwargs = {
                "train_folder": Path(self.data_path) / Path('train'),
                "valid_folder": Path(self.data_path) / Path('val'),
                "test_folder": Path(self.data_path) / Path('test')
            }

            # eliminate parameters that don't exist
            for k, v in data_kwargs:
                if not v.exists():
                    del data_kwargs[k]

            # raise error if no data was found
            if not data_kwargs:
                raise ValueError(
                    f"Could not find data in the following path: {self.data_path}. Are you sure training data is in the right location available?")

            # create data module with whatever data is available
            self._data = ImageClassificationData.from_folders(
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                **data_kwargs
            )

        # skip if in memory cache
        return self._data

    def train(self) -> None:
        """Start training using Lightning as workhorse"""

        print('train samples:', len(self.data.train_dataloader().dataset))
        print('valid samples:', len(self.data.val_dataloader().dataset))

        # create trainer and finetune
        trainer = Trainer(
            gpus=self.gpus,
            max_epochs=self.max_epochs
        )
        trainer.finetune(self.model, self.data, strategy='no_freeze')

    def predict(self, path:str):
        image_tensor = self.data_processing(path=path)
        # TODO: map prediction to class name
        # return model.forward(image_tensor)
