import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import DTD

from data_handling.augmentations import get_augmentations_from_config
from default_paths import DATA_DTD


class BaseDataModuleClass(LightningDataModule):
    def __init__(self, config: DictConfig, shuffle: bool = True) -> None:
        super().__init__()
        self.config = config
        self.shuffle = shuffle
        self.train_tsfm, self.val_tsfm = get_augmentations_from_config(
            config.data, use_model_preprocessing=config.model.predefined_preprocessing
        )
        self.image_size = (
            self.config.data.augmentations.center_crop
            if self.config.data.augmentations.center_crop != "None"
            else self.config.data.augmentations.resize
        )

        if not config.trainer.use_train_augmentations:
            self.train_tsfm = self.val_tsfm
        self.sampler = None

    def train_dataloader(self):
        if self.sampler is not None and self.shuffle:
            return DataLoader(
                self.dataset_train,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory,
                persistent_workers=False,
                batch_sampler=self.sampler,
            )
        return DataLoader(
            self.dataset_train,
            self.config.data.batch_size,
            shuffle=self.shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
        )

    @property
    def dataset_name(self):
        raise NotImplementedError

    def create_datasets(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError

    def get_irrelevant_ood_loader(self, prop_of_val=0.02):
        # Loading the data necessary for EBS
        ood_dataset = DTD(
            root=DATA_DTD, split="test", transform=self.val_tsfm, download=True
        )
        rng = np.random.default_rng(33)
        ood_dataset = Subset(
            ood_dataset,
            rng.choice(
                a=len(ood_dataset), size=int(prop_of_val * len(self.dataset_val))
            ),
        )
        print(len(ood_dataset))
        return DataLoader(
            ood_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
        )
