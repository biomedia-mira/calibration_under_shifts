'''
Main entry point for model training.
Simply run
python classification/train.py experiment='base_camelyon'
'''
from pathlib import Path
from omegaconf import OmegaConf
import hydra

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from classification.load_model_and_config import (
    get_modules,
    get_run_id_from_config,
    _clean_config_for_backward_compatibility,
)
from copy import deepcopy


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def train_model_main(config):
    print(config)

    config2 = deepcopy(config)
    _clean_config_for_backward_compatibility(config2)
    if (
        get_run_id_from_config(
            config2, allow_return_none_if_no_runs=True, return_running_jobs=True
        )
        is not None
    ):
        print(
            """
            A run already exists for this config. Skipping training.
            """
        )
        return

    pl.seed_everything(config.seed, workers=True)
    data_module, model_module = get_modules(config)

    wandb_logger = WandbLogger(save_dir="outputs", project=config.project_name)

    output_dir = Path(f"outputs/run_{wandb_logger.experiment.id}")  # type: ignore
    print("Saving to" + str(output_dir.absolute()))

    wandb_logger.watch(model_module, log="all", log_freq=100)

    wandb_logger.log_hyperparams(
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )

    callbacks = [LearningRateMonitor()]

    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename="{epoch}")
    callbacks.append(checkpoint_callback)

    checkpoint_callback_best = ModelCheckpoint(
        dirpath=output_dir,
        monitor=config.trainer.metric_to_monitor,
        mode=config.trainer.metric_to_monitor_mode,
        filename="best",
    )
    callbacks.append(checkpoint_callback_best)

    early_stopping = EarlyStopping(
        monitor=config.trainer.metric_to_monitor,
        mode=config.trainer.metric_to_monitor_mode,
        patience=round(config.trainer.patience_for_early_stop),
    )
    callbacks.append(early_stopping)

    precision = "32-true"
    torch.set_float32_matmul_precision("medium")
    if config.mixed_precision:
        precision = "16-mixed"
    n_gpus = (
        config.trainer.device
        if isinstance(config.trainer.device, int)
        else len(config.trainer.device)
    )
    trainer = pl.Trainer(
        deterministic="warn",
        accelerator="auto",
        devices=config.trainer.device,
        strategy="ddp_find_unused_parameters_true" if n_gpus > 1 else "auto",
        max_epochs=config.trainer.num_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision,
        fast_dev_run=config.is_unit_test_config,
        val_check_interval=(
            min(config.trainer.val_check_interval, len(data_module.train_dataloader()))
            if config.trainer.val_check_interval != "None"
            else None
        ),
    )

    trainer.fit(
        model_module,
        data_module,
    )

    trainer.validate(
        model_module, data_module, ckpt_path=trainer.checkpoint_callback.best_model_path
    )

    trainer.test(
        model_module, data_module, ckpt_path=trainer.checkpoint_callback.best_model_path
    )

    # run_inference(config)


if __name__ == "__main__":
    """
    Script to run one particular configuration.
    """
    torch.multiprocessing.set_sharing_strategy("file_system")
    train_model_main()
