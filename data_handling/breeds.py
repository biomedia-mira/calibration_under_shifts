from copy import deepcopy
from sklearn.model_selection import train_test_split
from data_handling.base import BaseDataModuleClass
from data_handling.breeds_helper import (
    make_living17,
    make_entity30,
    CustomImageNet,
)
from default_paths import BREEDS_INFO_DIR, DATA_IMAGENET
from torchvision.transforms import Compose
from torch.utils.data import Subset
import numpy as np


class BreedsDataModuleBase(BaseDataModuleClass):
    def create_datasets(self):
        ret = self.dataset_creation_fn(BREEDS_INFO_DIR, split="rand")
        self.superclasses, subclass_split, _ = ret
        train_subclasses, test_subclasses = subclass_split
        # because the API is stupid and requires Compose
        self.train_tsfm, self.val_tsfm = Compose([self.train_tsfm]), Compose(
            [self.val_tsfm]
        )
        dataset_source = CustomImageNet(
            DATA_IMAGENET,
            train_subclasses,
            transform_train=self.train_tsfm,
            transform_test=self.val_tsfm,
        )

        loaders_source = dataset_source.make_loaders(
            self.config.data.num_workers,
            self.config.data.batch_size,
            data_aug=self.shuffle,
            shuffle_val=False,
        )

        dev_loader_source, test_loader_source = loaders_source

        dev_dataset = dev_loader_source.dataset
        all_indices = np.arange(len(dev_dataset))
        train_idx, val_idx = train_test_split(all_indices, test_size=0.15)
        print(train_idx[:5], val_idx[:5])
        self.dataset_train = Subset(dev_dataset, train_idx)
        self.dataset_val = Subset(deepcopy(dev_dataset), val_idx)
        self.dataset_val.dataset.transform = self.val_tsfm
        self.dataset_test = test_loader_source.dataset

        dataset_target = CustomImageNet(
            DATA_IMAGENET,
            test_subclasses,
            transform_train=self.val_tsfm,
            transform_test=self.val_tsfm,
        )
        _, self.val_loader_target = dataset_target.make_loaders(
            self.config.data.num_workers, self.config.data.batch_size, shuffle_val=False
        )

        print(
            len(self.dataset_train),
            len(self.dataset_val),
            len(self.dataset_test),
            len(self.val_loader_target.dataset),
        )

    def get_evaluation_ood_dataloaders(self):
        return {"ood_test": self.val_loader_target}

    @property
    def num_classes(self):
        superclasses, _, _ = self.dataset_creation_fn(BREEDS_INFO_DIR, split="rand")
        return len(superclasses)


class Living17DataModule(BreedsDataModuleBase):
    @property
    def dataset_creation_fn(self):
        return make_living17

    @property
    def dataset_name(self):
        return "living17"


class Entity30DataModule(BreedsDataModuleBase):
    @property
    def dataset_creation_fn(self):
        return make_entity30

    @property
    def dataset_name(self):
        return "entity30"
