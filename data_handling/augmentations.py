from typing import Any, Callable, Tuple

import torch
import torchvision.transforms as tf
from omegaconf import DictConfig


class ExpandChannels:
    """
    Transform 1-channel into 3-channel image, by copying the channel 3 times.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(data, 3, dim=0)


class RandomSharpness:
    def __init__(self, sharp):
        self.sharp = sharp

    def __call__(self, x: torch.Tensor) -> Any:
        sharp_min = 1 - self.sharp
        sharp_max = 1 + self.sharp
        random_sharp = sharp_min + (sharp_max - sharp_min) * torch.rand(1)
        return tf.RandomAdjustSharpness(random_sharp)(x)


class ToTensorIfNecessary:
    def __call__(self, x) -> Any:
        if isinstance(x, torch.Tensor):
            return x
        return tf.ToTensor()(x)


def get_augmentations_from_config(
    config: DictConfig, use_model_preprocessing: bool
) -> Tuple[Callable, Callable]:
    """
    Return transformation pipeline as per config.
    """
    no_preprocess = (
        not use_model_preprocessing
    )  # for foundation models use the pre-defined model preprocessing, hence only load the training augmentations.
    if no_preprocess:
        transform_list, val_transforms = [ToTensorIfNecessary()], [
            ToTensorIfNecessary()
        ]
    else:
        transform_list, val_transforms = [], []
    if config.augmentations.random_crop != "None":
        transform_list.append(
            tf.RandomResizedCrop(
                config.augmentations.resize,
                scale=config.augmentations.random_crop,
                antialias=True,
            )
        )
        val_transforms.append(tf.Resize(config.augmentations.resize, antialias=True))

    if config.augmentations.resize != "None":
        transform_list.append(tf.Resize(config.augmentations.resize, antialias=True))
        val_transforms.append(tf.Resize(config.augmentations.resize, antialias=True))

    if config.augmentations.random_rotation != "None":
        transform_list.append(tf.RandomRotation(config.augmentations.random_rotation))
    if config.augmentations.horizontal_flip:
        transform_list.append(tf.RandomHorizontalFlip())
    if config.augmentations.vertical_flip:
        transform_list.append(tf.RandomVerticalFlip())
    if config.augmentations.random_color_jitter:
        transform_list.append(
            tf.ColorJitter(
                brightness=config.augmentations.random_color_jitter,
                contrast=config.augmentations.random_color_jitter,
                hue=(
                    0
                    if config.input_channels == 1
                    else config.augmentations.random_color_jitter
                ),
                saturation=(
                    0
                    if config.input_channels == 1
                    else config.augmentations.random_color_jitter
                ),
            )
        )

    if config.augmentations.random_erase_scale[0] > 0.0:
        transform_list.append(
            tf.RandomErasing(
                scale=[
                    config.augmentations.random_erase_scale[0],
                    config.augmentations.random_erase_scale[1],
                ]
            )
        )

    if config.augmentations.sharp > 0.0:
        transform_list.append(RandomSharpness(config.augmentations.sharp))

    if no_preprocess and config.augmentations.normalize.mean != "None":
        transform_list.append(
            tf.Normalize(
                config.augmentations.normalize.mean, config.augmentations.normalize.std
            )
        )
        val_transforms.append(
            tf.Normalize(
                config.augmentations.normalize.mean, config.augmentations.normalize.std
            )
        )

    if config.augmentations.center_crop != "None":
        transform_list.append(tf.CenterCrop(config.augmentations.center_crop))
        val_transforms.append(tf.CenterCrop(config.augmentations.center_crop))

    if no_preprocess and config.augmentations.expand_channels:
        transform_list.extend([ExpandChannels()])
        val_transforms.extend([ExpandChannels()])

    if no_preprocess:
        return tf.Compose(transform_list), tf.Compose(val_transforms)
    else:
        return tf.Compose(transform_list), tf.Compose(val_transforms)


class GaussianBlur(tf.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 0.7
    ):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = tf.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)
