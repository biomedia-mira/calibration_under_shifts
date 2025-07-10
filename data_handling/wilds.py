from sklearn.model_selection import train_test_split
from data_handling.base import BaseDataModuleClass
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader
from default_paths import DATA_WILDS_ROOT
from torch.utils.data import DataLoader
from torchvision.datasets import DTD
import numpy as np
from copy import deepcopy


class WILDSBase(BaseDataModuleClass):
    def create_datasets(self) -> None:
        raise NotImplementedError("setup() should be implemented in child class")

    def get_evaluation_ood_dataloaders(self):
        return {
            "ood_val": get_eval_loader(
                "standard",
                self.ood_val_dataset,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
            ),
            "ood_test": get_eval_loader(
                "standard",
                self.ood_test_dataset,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
            ),
        }

    @property
    def dataset_name(self):
        return self.wilds_class_name


class WILDSCameLyon17(WILDSBase):
    def create_datasets(self) -> None:
        self.dataset = get_dataset(
            dataset="camelyon17", download=True, root_dir=DATA_WILDS_ROOT
        )

        self.dataset_dev = self.dataset.get_subset("train", transform=self.val_tsfm)
        train_idx, val_idx = train_test_split(self.dataset_dev.indices, test_size=0.1)
        self.dataset_train = deepcopy(self.dataset_dev)
        self.dataset_train.indices = train_idx
        self.dataset_train.transform = self.train_tsfm
        self.dataset_val = self.dataset_dev
        self.dataset_val.indices = val_idx
        self.dataset_test = self.dataset.get_subset("id_val", transform=self.val_tsfm)
        self.ood_test_dataset = self.dataset.get_subset("test", transform=self.val_tsfm)
        self.ood_val_dataset = self.dataset.get_subset("val", transform=self.val_tsfm)
        print(
            len(self.dataset_train),
            len(self.dataset_val),
            len(self.dataset_test),
            len(self.ood_val_dataset),
            len(self.ood_test_dataset),
        )

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def wilds_class_name(self):
        return "wilds_camelyon"


class WILDSiCam(WILDSBase):
    def create_datasets(self) -> None:
        self.dataset = get_dataset(
            dataset="iwildcam", download=True, root_dir=DATA_WILDS_ROOT
        )
        self.dataset_train = self.dataset.get_subset("train", transform=self.train_tsfm)
        self.dataset_val = self.dataset.get_subset("id_val", transform=self.val_tsfm)
        self.dataset_test = self.dataset.get_subset("id_test", transform=self.val_tsfm)
        self.ood_val_dataset = self.dataset.get_subset("val", transform=self.val_tsfm)
        self.ood_test_dataset = self.dataset.get_subset("test", transform=self.val_tsfm)
        print(self.dataset._n_classes)
        print(
            len(self.dataset_train),
            len(self.dataset_val),
            len(self.dataset_test),
            len(self.ood_val_dataset),
            len(self.ood_test_dataset),
        )

    @property
    def num_classes(self) -> int:
        return 182

    @property
    def wilds_class_name(self):
        return "wilds_icam"


DOMAIN_NET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


class DomainNet(WILDSBase):
    def create_datasets(self) -> None:
        self.dataset = get_dataset(
            dataset="domainnet",
            download=True,
            root_dir=DATA_WILDS_ROOT,
            source_domain=self.config.data.train_domain,
        )
        self.dataset_dev = self.dataset.get_subset("train", transform=self.val_tsfm)
        train_idx, val_idx = train_test_split(self.dataset_dev.indices, test_size=0.1)
        self.dataset_train = deepcopy(self.dataset_dev)
        self.dataset_train.indices = train_idx
        self.dataset_train.transform = self.train_tsfm
        self.dataset_val = self.dataset_dev
        self.dataset_val.indices = val_idx
        self.dataset_test = self.dataset.get_subset("id_test", transform=self.val_tsfm)
        print(len(self.dataset_train), len(self.dataset_val), len(self.dataset_test))

    def get_evaluation_ood_dataloaders(self):
        test_domains = np.setdiff1d(
            np.asarray(DOMAIN_NET_DOMAINS), np.array([self.config.data.train_domain])
        )
        print(test_domains)
        loaders = {}
        for domain in test_domains:
            dataset = get_dataset(
                dataset="domainnet",
                download=True,
                root_dir=DATA_WILDS_ROOT,
                source_domain=self.config.data.train_domain,
                target_domain=domain,
            )
            test_target = dataset.get_subset("test", transform=self.val_tsfm)
            test_loader = DataLoader(
                test_target,
                self.config.data.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
            )
            loaders[f"{domain}_test"] = test_loader
            print(f"{domain}_test", len(test_target))
        return loaders

    @property
    def num_classes(self) -> int:
        return 345

    @property
    def wilds_class_name(self):
        return "domainnet"
