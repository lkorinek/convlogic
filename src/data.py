import math
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T


class ConvLogicDataModule(pl.LightningDataModule):
    """
    Data module for MNIST and CIFAR-10 with threshold-based transforms.
    Splits into train, validation, and test sets, and produces DataLoaders.
    """

    def __init__(
        self,
        dataset_name="cifar10-3",
        data_dir="../../data/",
        batch_size=128,
        num_workers=2,
        threshold_levels=None,
        train_val_split=0.9,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = os.path.abspath(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        if threshold_levels is not None:
            self.threshold_levels = threshold_levels
        else:
            if "-" in dataset_name:
                try:
                    self.threshold_levels = int(dataset_name.split("-")[-1])
                except ValueError:
                    raise ValueError(f"Invalid threshold suffix: {dataset_name}")
            else:
                self.threshold_levels = 1

        self.transforms = self._build_transforms()

    def _build_transforms(self):
        def threshold_transform(x):
            thresholds = [(x > ((i + 1) / (self.threshold_levels + 1))).float() for i in range(self.threshold_levels)]
            return torch.cat(thresholds, dim=0)

        base_transforms = [T.ToTensor(), T.Lambda(threshold_transform)]

        return T.Compose(base_transforms)

    def setup(self, stage: str = None):
        if self.dataset_name.startswith("cifar10"):
            dataset_class = datasets.CIFAR10
        elif self.dataset_name.startswith("mnist"):
            dataset_class = datasets.MNIST
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        full_train_set = dataset_class(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transforms,
        )
        self.test_set = dataset_class(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transforms,
        )

        train_size = math.ceil(self.train_val_split * len(full_train_set))
        valid_size = len(full_train_set) - train_size
        self.train_set, self.valid_set = random_split(full_train_set, [train_size, valid_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    @property
    def input_channels(self):
        """
        Returns the number of input channels after thresholding:
        - MNIST: threshold_levels x 1
        - CIFAR-10: threshold_levels x 3
        """
        if self.dataset_name.startswith("mnist"):
            return self.threshold_levels
        elif self.dataset_name.startswith("cifar10"):
            return 3 * self.threshold_levels
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
