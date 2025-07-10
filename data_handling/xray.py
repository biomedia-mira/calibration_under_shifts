from pathlib import Path
from typing import Callable, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import ToTensor, Resize, CenterCrop, ToPILImage
from data_handling.base import BaseDataModuleClass
from data_handling.caching import SharedCache
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CHEXPERT_ROOT = Path('/vol/biodata/data/chest_xray/CheXpert-v1.0')
MIMIC_ROOT = Path("/vol/biodata/data/chest_xray/mimic-cxr-jpg-224")

class CheXpertDataModule(BaseDataModuleClass):
    def create_datasets(self):
        label_col = self.config.data.label
        df = pd.read_csv(CHEXPERT_ROOT / "meta" / "train.csv")
        df.fillna(0, inplace=True)  # assume no mention is like negative
        df = df.loc[df["AP/PA"].isin(["AP", "PA"])]
        df = df.loc[df[self.config.data.label] != -1]  # remove the uncertain cases
        df["PatientID"] = df["Path"].apply(
            lambda x: int(Path(x).parent.parent.stem[-5:])
        )
        patient_id = df["PatientID"].unique()
        train_val_id, test_id = train_test_split(
            patient_id, test_size=0.2, random_state=33
        )
        train_id, val_id = train_test_split(
            train_val_id, test_size=0.15, random_state=33
        )

        self.dataset_train = CheXpertDataset(
            df=df.loc[df.PatientID.isin(train_id)],
            transform=self.train_tsfm,
            cache=self.config.data.cache,
            label_col=label_col,
        )

        self.dataset_val = CheXpertDataset(
            df=df.loc[df.PatientID.isin(val_id)],
            transform=self.val_tsfm,
            cache=self.config.data.cache,
            label_col=label_col,
        )

        self.dataset_test = CheXpertDataset(
            df=df.loc[df.PatientID.isin(test_id)],
            transform=self.val_tsfm,
            cache=False,
            label_col=label_col,
        )
        print(len(self.dataset_train), len(self.dataset_val), len(self.dataset_test))

    @property
    def dataset_name(self):
        return "chexpert"

    @property
    def num_classes(self):
        return 2

    def get_evaluation_ood_dataloaders(self):
        evaluation_loaders = {}
        df = pd.read_csv(MIMIC_ROOT / "meta" / "mimic-cxr-2.0.0-chexpert.csv")
        df.fillna(0, inplace=True)
        df_meta = pd.read_csv(MIMIC_ROOT / "meta" / "mimic-cxr-2.0.0-metadata.csv")
        df_full = pd.merge(df, df_meta, how="inner", on=["subject_id", "study_id"])
        df_full = df_full.loc[df_full[self.config.data.label] != -1]
        df_full = df_full.loc[df_full["ViewPosition"].isin(["AP", "PA"])]
        # Full MIMIC dataset too big for testing. Take 25000 images only.
        df_full = df_full.sample(n=25000, replace=False, random_state=self.config.seed)

        mimic_dataset = MIMICDataset(
            df=df_full,
            label_col=self.config.data.label,
            transform=self.val_tsfm,
            cache=False,
        )

        loader = DataLoader(
            mimic_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
        )
        print(len(mimic_dataset))
        evaluation_loaders["MIMIC"] = loader
        return evaluation_loaders


class CheXpertDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        transform: Callable,
        cache: bool = False,
    ):
        super().__init__()
        print(f"Len dataset {len(df)}")
        df.fillna(0, inplace=True)
        self.labels = df[label_col].astype(int).values
        self.img_paths = df.Path.values
        self.cache = cache
        self.transform = transform

        if cache:
            self.cache = SharedCache(
                size_limit_gib=36,
                dataset_len=self.img_paths.shape[0],
                data_dims=[3, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.img_paths)

    def read_image(self, idx):
        img = Image.open(CHEXPERT_ROOT / '..' / self.img_paths[idx])
        img = CenterCrop(224)(Resize(224, antialias=True)(img))
        img = img.convert("RGB")
        img = ToTensor()(img)
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
        sample["x"] = self.transform(img).float()
        return sample


class MIMICDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        transform: Callable,
        cache: bool = False,
    ):
        super().__init__()
        print(f"Len dataset {len(df)}")
        self.labels = df[label_col].astype(int).values
        self.study_ids = df["study_id"].values
        self.dicom_ids = df["dicom_id"].values
        self.patient_id = df["subject_id"].values
        self.cache = cache
        self.transform = transform

        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=self.dicom_ids.shape[0],
                data_dims=[3, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.dicom_ids)

    def read_image(self, idx):
        pid = str(self.patient_id[idx])
        img = Image.open(
            Path(MIMIC_ROOT / "files")
            / f"p{pid[:2]}"
            / f"p{pid}"
            / f"s{self.study_ids[idx]}"
            / f"{self.dicom_ids[idx]}.jpg"
        )
        img = CenterCrop(224)(Resize(224, antialias=True)(img))
        img = img.convert("RGB")
        img = ToTensor()(img)
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
        sample["x"] = self.transform(img).float()
        return sample
