from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from sklearn.model_selection import train_test_split
from data_handling.base import BaseDataModuleClass
from data_handling.caching import SharedCache
from torchvision.transforms import Resize, ToPILImage
from torch.utils.data import DataLoader, Dataset, Subset

EMBED_ROOT = "/vol/biomedic3/data/EMBED"


domain_maps = {
    "HOLOGIC, Inc.": 0,
    "GE MEDICAL SYSTEMS": 1,
    "FUJIFILM Corporation": 2,
    "GE HEALTHCARE": 3,
    "Lorad, A Hologic Company": 4,
}

tissue_maps = {"A": 0, "B": 1, "C": 2, "D": 3}
modelname_map = {
    "Selenia Dimensions": 0,
    "Senographe Essential VERSION ADS_53.40": 5,
    "Senographe Essential VERSION ADS_54.10": 5,
    "Senograph 2000D ADS_17.4.5": 2,
    "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3,
    "Clearview CSm": 4,
    "Senographe Pristina": 1,
}

modelname_simplified = {
    "Selenia Dimensions": "Selenia Dimensions",
    "Senographe Essential VERSION ADS_53.40": "Senographe Essential",
    "Senographe Essential VERSION ADS_54.10": "Senographe Essential",
    "Senograph 2000D ADS_17.4.5": "Senograph 2000D",
    "Senograph 2000D ADS_17.5": "Senograph 2000D",
    "Lorad Selenia": "Lorad Selenia",
    "Clearview CSm": "Clearview CSm",
    "Senographe Pristina": "Senographe Pristina",
}


def preprocess_breast(image_path, target_size):
    """
    Loads the image performs basic background removal around the breast.
    Works for text but not for objects in contact with the breast (as it keeps the
    largest non background connected component.)
    """
    image = cv2.imread(str(image_path))

    if image is None:
        # sometimes bug in reading images with cv2
        from skimage.util import img_as_ubyte

        image = io.imread(image_path)
        gray = img_as_ubyte(image.astype(np.uint16))
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

    # Connected components with stats.
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=4
    )

    # Find the largest non background component.
    # Note: range() starts from 1 since 0 is the background label.
    max_label, _ = max(
        [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
        key=lambda x: x[1],
    )
    mask = output == max_label
    img = torch.tensor((gray * mask) / 255.0).unsqueeze(0).float()
    img = Resize(target_size, antialias=True)(img)
    return img


def get_embed_csv():
    image_dir = EMBED_ROOT / Path("images/png/1024x768")
    try:
        mydf = pd.read_csv(Path(__file__).parent / "joined_simple.csv")
    except FileNotFoundError:
        print(
            """
            For running EMBED code you need to first generate the csv
            file used for this study in csv_generation_code/generate_embed_csv.ipynb
            """
        )
        raise FileNotFoundError(
            """
            For running EMBED code you need to first generate the csv
            file used for this study in csv_generation_code/generate_embed_csv.ipynb
            """
        )

    mydf["shortimgpath"] = mydf["image_path"]
    mydf["image_path"] = mydf["image_path"].apply(lambda x: image_dir / str(x))

    mydf["manufacturer_domain"] = mydf.Manufacturer.apply(lambda x: domain_maps[x])

    # convert tissueden to trainable label
    mydf["tissueden"] = mydf.tissueden.apply(lambda x: tissue_maps[x])

    mydf["SimpleModelLabel"] = mydf.ManufacturerModelName.apply(
        lambda x: modelname_map[x]
    )
    print(mydf.SimpleModelLabel.value_counts())
    mydf["ViewLabel"] = mydf.ViewPosition.apply(lambda x: 0 if x == "MLO" else 1)

    mydf["CviewLabel"] = mydf.FinalImageType.apply(lambda x: 0 if x == "2D" else 1)

    mydf = mydf.dropna(
        subset=[
            "tissueden",
            "SimpleModelLabel",
            "ViewLabel",
            "image_path",
        ]
    )

    return mydf


class EmbedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: torch.nn.Module,
        target_size,
        label="tissueden",
        cache: bool = True,
    ) -> None:
        self.imgs_paths = df.image_path.values
        self.shortpaths = df.shortimgpath.values
        self.labels = df[label].values
        print(df[label].value_counts())

        self.transform = transform
        self.target_size = target_size
        self.views = df.ViewLabel.values
        self.scanner = df.SimpleModelLabel.values
        self.cview = df.FinalImageType.apply(lambda x: 0 if x == "2D" else 1).values
        self.densities = df.tissueden.values
        data_dims = [1, self.target_size[0], self.target_size[1]]
        if cache:
            self.cache = SharedCache(
                size_limit_gib=96,
                dataset_len=self.labels.shape[0],
                data_dims=data_dims,
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __getitem__(self, index) -> Any:
        if self.cache is not None:
            # retrieve data from cache if it's there
            img = self.cache.get_slot(index)
            # x will be None if the cache slot was empty or OOB
            if img is None:
                img = preprocess_breast(self.imgs_paths[index], self.target_size)
                self.cache.set_slot(index, img, allow_overwrite=True)  # try to cache x
        else:
            img = preprocess_breast(self.imgs_paths[index], self.target_size)
        # this needs to be the case for applying the foundation models.
        img = ToPILImage()(img)
        img = img.convert("RGB")
        sample = {}
        sample["cview"] = self.cview[index]
        sample["shortpath"] = str(self.shortpaths[index])
        sample["view"] = self.views[index]
        sample["density"] = torch.nn.functional.one_hot(
            torch.tensor(self.densities[index]).long(), num_classes=4
        ).detach()
        sample["y"] = self.labels[index]
        sample["scanner_int"] = self.scanner[index]
        sample["scanner"] = torch.nn.functional.one_hot(
            torch.tensor(self.scanner[index]).long(), num_classes=6
        ).detach()

        img = self.transform(img)
        sample["x"] = img.float()
        return sample

    def __len__(self):
        return self.labels.shape[0]


class EmbedDataModule(BaseDataModuleClass):
    @property
    def dataset_name(self) -> str:
        return "EMBED"

    def create_datasets(self) -> None:
        full_df = get_embed_csv()

        # Use only Selenia Dimension 2D images for training.
        id_df = full_df.loc[full_df.FinalImageType == "2D"]
        id_df = id_df.loc[id_df.ManufacturerModelName == "Selenia Dimensions"]

        dev_id, test_id = train_test_split(
            id_df.empi_anon.unique(), test_size=0.20, random_state=33
        )

        train_id, val_id = train_test_split(dev_id, test_size=0.20, random_state=33)

        self.target_size = self.config.data.augmentations.resize
        self.dataset_train = EmbedDataset(
            df=id_df.loc[id_df.empi_anon.isin(train_id)],
            transform=self.train_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            cache=self.config.data.cache,
        )

        self.dataset_val = EmbedDataset(
            df=id_df.loc[id_df.empi_anon.isin(val_id)],
            transform=self.val_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            cache=self.config.data.cache,
        )

        self.dataset_test = EmbedDataset(
            df=id_df.loc[id_df.empi_anon.isin(test_id)],
            transform=self.val_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            cache=False,
        )
        print(len(self.dataset_train), len(self.dataset_val), len(self.dataset_test))
        # OOD is all other manufacturers, or the C-view images of Selenia Dimensions
        ood_df = full_df.loc[
            (full_df.ManufacturerModelName != "Selenia Dimensions")
            | (full_df.FinalImageType != "2D")
        ]
        ood_df["Domain"] = (
            ood_df.ManufacturerModelName.apply(lambda x: modelname_simplified[x])
            + ood_df.FinalImageType
        )
        self.ood_df = ood_df
        print(ood_df.Domain.value_counts())

    @property
    def num_classes(self) -> int:
        match self.config.data.label:
            case "tissueden":
                return 4
            case "SimpleModelLabel":
                return 5
            case "ViewLabel":
                return 2
            case "CviewLabel":
                return 2
            case _:
                raise ValueError

    def get_evaluation_ood_dataloaders(self):
        evaluation_loaders = {}
        for domain in self.ood_df.Domain.unique():
            domain_dataset = EmbedDataset(
                df=self.ood_df.loc[self.ood_df.Domain == domain],
                transform=self.val_tsfm,
                target_size=self.target_size,
                label=self.config.data.label,
                cache=False,
            )
            loader = DataLoader(
                domain_dataset,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
                shuffle=False,
            )
            evaluation_loaders[domain] = loader
        return evaluation_loaders
