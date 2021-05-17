import os
import json
import timm
import torch

from copy import deepcopy
from pathlib import Path
from flash import Trainer
import pytorch_lightning as pl

from fdetection.data import DataProcessing
from flash.vision import ImageClassificationData, ImageClassifier
from typing import Optional


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
    def __init__(self, train_data_path:Optional[str] = None, backbone: str = "resnet18", batch_size: int = 32, num_workers: int = 4,
                 seed:int = 1234, learning_rate:float = 0.001, gpus:int = 0, max_epochs: int = 1, inference: bool = False, 
                 checkpoint_path: Optional[str] = None):
        
        self.train_data_path = train_data_path
        self.backbone = backbone
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

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

        # load model to memory if we are requesting that
        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            self.labels_path = "labels.json"
            self._load_class_labels()
            self.inference_model = ImageClassifier.load_from_checkpoint(checkpoint_path)

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
                                optimizer=torch.optim.Adam,
                                learning_rate=self.learning_rate)

        return model

    @property
    def data(self) -> ImageClassificationData:
        """Creates DataLoaders if not available"""
        if not hasattr(self, "_data"):
            # define folder location
            data_kwargs = {
                "train_folder": Path(self.train_data_path) / Path('train'),
                "valid_folder": Path(self.train_data_path) / Path('val'),
                "test_folder": Path(self.train_data_path) / Path('test')
            }

            # eliminate parameters that don't exist
            for k, v in deepcopy(data_kwargs).items():
                if not v.exists():
                    del data_kwargs[k]

            # raise error if no data was found
            if not data_kwargs:
                raise ValueError(
                    f"Could not find data in the following path: {self.train_data_path}. Are you sure training data is in the right location available?")

            # create data module with whatever data is available
            self._data = ImageClassificationData.from_folders(
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                **data_kwargs
            )

        # skip if memory cache available
        return self._data
    
    def _load_class_labels(self) -> None:
        with open(self.labels_path) as f:
            self.class_labels = json.load(f)["labels"]

        print(f"Loaded {len(self.class_labels)} class labels: {self.class_labels}")

    def train(self) -> None:
        """Start training using Lightning as workhorse"""

        # print usefult statistics
        print('train samples:', len(self.data.train_dataloader().dataset))
        print('valid samples:', len(self.data.val_dataloader().dataset))

        # create trainer and finetune
        trainer = Trainer(
            gpus=self.gpus,
            max_epochs=self.max_epochs
        )
        trainer.finetune(self.model, self.data, strategy='no_freeze')

    def predict(self, path:str):
        """Make predictions from a single file."""
        tensors = self.data_processing.process(path=path)
        for tensor in tensors:
            predictions = self.inference_model.predict([tensor])
            print(f"Predicted class: {self.class_labels[predictions[0]]}")
