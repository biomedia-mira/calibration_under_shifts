from typing import Dict, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_handling.base import BaseDataModuleClass
from data_handling.caching import SharedCache
from torchvision.transforms import ToTensor, Resize, CenterCrop, ToPILImage
from torch.utils.data import DataLoader, Dataset
from default_paths import ROOT


class RetinaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable = torch.nn.Identity(),
        cache: bool = False,
    ):
        super().__init__()
        print(f"Len {len(df)}")
        self.sites = df.site.astype(int).values
        self.labels = df.diagnosis.astype(int).values
        self.img_paths = df.img_path.values

        self.cache = cache
        self.transform = transform

        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=self.img_paths.shape[0],
                data_dims=[3, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.img_paths)

    def read_image(self, idx):
        img = Image.open(self.img_paths[idx])
        img = CenterCrop(224)(Resize(224, antialias=True)(ToTensor()(img)))
        return img

    def __getitem__(self, idx: int) -> Dict:
        if self.cache is not None:
            img = self.cache.get_slot(idx)
            if img is None:
                img = self.read_image(idx)
                self.cache.set_slot(idx, img, allow_overwrite=True)

        else:
            img = self.read_image(idx)

        # Can only be cached as tensor but needs to be PIL for pretrained processing functions
        img = ToPILImage()(img)
        sample = {}
        sample["y"] = self.labels[idx]
        # sample["dr"] = self.sublabels[idx]
        sample["site"] = self.sites[idx]

        img = self.transform(img).float()

        sample["x"] = img

        return sample


class NewRetinaDataModule(BaseDataModuleClass):
    def create_datasets(self):
        train_df = pd.read_csv(ROOT / "data/retina_eyepacs_train.csv")
        val_df = pd.read_csv(ROOT / "data/retina_eyepacs_val.csv")
        test_df = pd.read_csv(ROOT / "data/retina_all_test.csv")

        self.TRAIN_SITE = 3
        self.OOD_SITES = [1, 2]
        train_df = train_df.loc[train_df.site == self.TRAIN_SITE]
        val_df = val_df.loc[val_df.site == self.TRAIN_SITE]
        test_df_id = test_df.loc[test_df.site == self.TRAIN_SITE]
        self.ood_df = test_df.loc[test_df.site.isin(self.OOD_SITES)]

        self.dataset_train = RetinaDataset(
            df=train_df,
            transform=self.train_tsfm,
            cache=self.config.data.cache,
        )

        self.dataset_val = RetinaDataset(
            df=val_df,
            transform=self.val_tsfm,
            cache=self.config.data.cache,
        )

        self.dataset_test = RetinaDataset(
            df=test_df_id,
            transform=self.val_tsfm,
        )

        print(len(self.dataset_train), len(self.dataset_val), len(self.dataset_test))

    @property
    def dataset_name(self):
        return "retina"

    @property
    def num_classes(self):
        return 5

    def get_evaluation_ood_dataloaders(self):
        evaluation_loaders = {}
        for domain in self.ood_df.site.unique():
            domain_dataset = RetinaDataset(
                df=self.ood_df.loc[self.ood_df.site == domain],
                transform=self.val_tsfm,
            )
            loader = DataLoader(
                domain_dataset,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
                shuffle=False,
            )
            evaluation_loaders[str(domain)] = loader
            print(domain, len(domain_dataset))
        return evaluation_loaders
