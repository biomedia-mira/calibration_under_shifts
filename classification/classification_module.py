'''
Generic image classification pytorch lightning module.
'''
import io
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Any
import timm

from torchmetrics.functional.classification import multiclass_calibration_error

from classification.model_wrappers import ClipWrapper, OpenClipModelWrapper


class ClassificationModule(pl.LightningModule):
    """
    A generic PL module for classification
    """

    def __init__(
        self,
        config,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.config = config
        self.loss = config.trainer.loss
        self.model = self.get_model()
        self.criterion = get_loss_function(
            weights=config.data.weights,
            label_smoothing=config.trainer.label_smoothing,
            entropy_regularisation=config.trainer.entropy_regularisation,
            module=self,
            focal_loss=config.trainer.use_focal_loss,
            focal_loss_gamma=config.trainer.focal_loss_gamma,
        )
        self.scheduler = config.trainer.scheduler
        self.save_hyperparameters()
        self.get_all_features = False
        self.train_transform = None
        self.val_transform = None

    def common_step(self, batch, prefix: str, batch_idx: int) -> Any:  # type: ignore
        if isinstance(batch, dict):
            data, target = batch["x"], batch["y"]
        else:
            data, target = batch[0], batch[1]
        features = self.model.forward_features(data)
        output = self.model.forward_head(features)
        loss = self.criterion(output, target)
        self.log(f"{prefix}/loss", loss)
        probas = torch.softmax(output, 1)

        if self.training and data.ndim == 4:
            if batch_idx == 0 and self.current_epoch < 2:
                data = data.cpu().numpy()
                self._plot_image_and_log_img_grid(
                    data, target.cpu().numpy(), "Train/inputs"
                )
                print(data.max())
        return loss, probas, target

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        if isinstance(batch, dict):
            data, target = batch["x"], batch["y"]
        else:
            data, target = batch[0], batch[1]
        last_feats, feats = self.model.forward_intermediates(data, indices=5)
        output = self.model.forward_head(last_feats)
        feats = [torch.nn.AdaptiveAvgPool2d((1, 1))(x).flatten(1) for x in feats]
        probas = torch.softmax(output, 1)
        return {"probas": probas, "targets": target, "logits": output, "feats": feats}

    def training_step(self, batch: Any, batch_idx: int) -> Any:  # type: ignore
        loss, probas, targets = self.common_step(
            batch, prefix="Train", batch_idx=batch_idx
        )
        self.train_probas.append(probas.detach().cpu())
        self.train_targets.append(targets.detach().cpu())
        assert not torch.any(loss.isnan())
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:  # type: ignore
        _, probas, targets = self.common_step(batch, prefix="Val", batch_idx=batch_idx)
        self.val_probas.append(probas.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    def test_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:  # type: ignore
        _, probas, targets = self.common_step(batch, prefix="Test", batch_idx=batch_idx)
        self.test_probas.append(probas.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

    def on_train_epoch_start(self) -> None:
        self.train_probas = []
        self.train_targets = []

    def _compute_metrics_at_epoch_end(
        self, targets: torch.Tensor, probas: torch.Tensor, prefix: str
    ):
        preds = torch.argmax(probas, 1)
        try:
            if self.num_classes == 2:
                self.log(
                    f"{prefix}/AUROC",
                    roc_auc_score(targets, probas[:, 1]),
                    sync_dist=True,
                )
            else:
                self.log(
                    f"{prefix}/AUROC",
                    roc_auc_score(targets, probas, average="macro", multi_class="ovr"),
                    sync_dist=True,
                )
        except ValueError:
            pass
        self.log(f"{prefix}/Accuracy", accuracy_score(targets, preds), sync_dist=True)
        self.log(
            f"{prefix}/BalAccuracy",
            balanced_accuracy_score(targets, preds),
            sync_dist=True,
        )
        self.log(
            f"{prefix}/ECE",
            multiclass_calibration_error(probas, targets, num_classes=self.num_classes),
            sync_dist=True,
        )

    def on_train_epoch_end(self, unused=None) -> None:
        if len(self.train_targets) > 0:
            targets, probas = torch.cat(self.train_targets), torch.cat(
                self.train_probas
            )
            self._compute_metrics_at_epoch_end(targets, probas, "Train")
        self.train_probas = []
        self.train_targets = []

    def on_validation_epoch_start(self) -> None:
        self.val_probas = []
        self.val_targets = []

    def on_validation_epoch_end(self, unused=None) -> None:
        targets, probas = torch.cat(self.val_targets), torch.cat(self.val_probas)
        self._compute_metrics_at_epoch_end(targets, probas, "Val")
        self.val_probas = []
        self.val_targets = []

    def on_test_epoch_start(self) -> None:
        self.test_probas = []
        self.test_targets = []

    def on_test_epoch_end(self, unused=None) -> None:
        targets, probas = torch.cat(self.test_targets), torch.cat(self.test_probas)
        self._compute_metrics_at_epoch_end(targets, probas, "Test")
        self.test_probas = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.trainer.lr,
            weight_decay=self.config.trainer.weight_decay,
        )
        match self.scheduler:
            case "reduce_lr_on_plateau":
                scheduler = {
                    "scheduler": ReduceLROnPlateau(
                        optimizer,
                        patience=self.config.trainer.patience_for_scheduler,
                        mode=self.config.trainer.metric_to_monitor_mode,
                        min_lr=1e-5,
                    ),
                    "monitor": self.config.trainer.metric_to_monitor,
                }
                return [optimizer], [scheduler]
            case "cosine":
                scheduler = {
                    "scheduler": CosineAnnealingLR(
                        optimizer, T_max=self.config.trainer.num_epochs, eta_min=1e-5
                    )
                }
                return [optimizer], [scheduler]
            case "constant":
                return optimizer
            case _:
                return ValueError(
                    f"Only reduce_lr_on_plateau and constant schedulers implemented. You specified {self.scheduler}"
                )


    def get_model(self) -> torch.nn.Module:
        if self.config.model.encoder_name == "openclip":
            model = OpenClipModelWrapper(
                checkpoint_name=self.config.model.checkpoint_name,
                num_classes=self.num_classes,
                freeze_encoder=self.config.model.freeze_encoder,
            )
            self.preprocess = model.preprocess
            return model
        elif self.config.model.encoder_name == "timm":
            model = timm.create_model(
                self.config.model.checkpoint_name, pretrained=True
            )
            if hasattr(model.head, "reset"):
                model.head.reset(num_classes=self.num_classes)
            elif isinstance(model.head, torch.nn.Linear) or isinstance(
                model.head, torch.nn.Identity
            ):
                model.head = torch.nn.Linear(
                    in_features=model.embed_dim, out_features=self.num_classes
                )
            else:
                raise AttributeError
            self.preprocess = timm.data.create_transform(
                **timm.data.resolve_data_config(model.pretrained_cfg), is_training=False
            )
            return model
        elif self.config.model.encoder_name == "clipvision":
            model = ClipWrapper(
                num_classes=self.num_classes,
                freeze_encoder=self.config.model.freeze_encoder,
            )
            self.preprocess = model.preprocess
            return model
        else:
            try:
                target_size = (
                    self.config.data.augmentations.center_crop
                    if self.config.data.augmentations.center_crop != "None"
                    else self.config.data.augmentations.resize
                )
                return timm.create_model(
                    self.config.model.encoder_name,
                    pretrained=self.config.model.pretrained,
                    num_classes=self.num_classes,
                    img_size=target_size,
                    in_chans=self.config.data.input_channels,
                )

            except TypeError:
                return timm.create_model(
                    self.config.model.encoder_name,
                    pretrained=self.config.model.pretrained,
                    num_classes=self.num_classes,
                    in_chans=self.config.data.input_channels,
                )

    def _plot_image_and_log_img_grid(self, data: np.ndarray, y: np.ndarray, tag: str):
        f, ax = plt.subplots(2, 5, figsize=(15, 5))

        if data.ndim == 5:
            for i in range(min(5, data.shape[0])):
                for j in range(2):
                    img = np.transpose(data[i, j], [1, 2, 0])
                    img = (img - img.min()) / (img.max() - img.min())
                    (
                        ax[j, i].imshow(img)
                        if img.shape[-1] == 3
                        else ax[j, i].imshow(img, cmap="gray")
                    )
                    if y is not None:
                        ax[j, i].set_title(y[i])
                    ax[j, i].axis("off")
        else:
            ax = ax.ravel()
            for i in range(min(10, data.shape[0])):
                img = np.transpose(data[i], [1, 2, 0])
                img = (img - img.min()) / (img.max() - img.min())
                (
                    ax[i].imshow(img)
                    if img.shape[-1] == 3
                    else ax[i].imshow(img, cmap="gray")
                )
                if y is not None:
                    ax[i].set_title(y[i])
                ax[i].axis("off")

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")

        im = Image.open(img_buf)
        self.logger.experiment.log({tag: wandb.Image(im)})
        img_buf.close()
        plt.close()


def get_loss_function(
    weights,
    label_smoothing,
    entropy_regularisation,
    module,
    focal_loss=False,
    focal_loss_gamma=None,
):
    weights = torch.tensor(weights) if weights != "None" else None

    if focal_loss:

        def loss_function(logits, target):
            probas = torch.softmax(logits, 1)
            return compute_focal_loss(target, probas, gamma=focal_loss_gamma)

    else:
        base_function = torch.nn.CrossEntropyLoss(
            weight=weights, label_smoothing=label_smoothing, reduction="none"
        )

        def loss_function(logits, target, log=False):
            ce_result = base_function(logits, target)
            if entropy_regularisation > 0:
                probas = torch.softmax(logits, 1)
                neg_entropy = (probas * torch.log(probas + 1e-16)).sum(1)
                if log:
                    module.log("Train/CE", ce_result)
                    module.log("Train/penalty", entropy_regularisation * neg_entropy)
                return (ce_result + entropy_regularisation * neg_entropy).mean()
            return ce_result.mean()

    return loss_function


def compute_focal_loss(targets, probas, gamma=3):
    one_hot_labels = torch.nn.functional.one_hot(targets, num_classes=probas.shape[1])
    pt = (probas * one_hot_labels).sum(1)
    # FLSD 53 approach from Mukhoti et al.
    if gamma == -53:
        gamma = torch.where(pt > 0.2, 3, 5)
    elif gamma < 0:
        raise ValueError("Gamma supposed to be > 0, for FLSD enter -53")
    fl = -((1 - pt) ** gamma * torch.log(pt + 1e-16))
    return fl.mean()