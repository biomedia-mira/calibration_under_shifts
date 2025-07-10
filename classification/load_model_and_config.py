from typing import Dict
import hydra
from omegaconf import OmegaConf
import wandb

from classification.classification_module import ClassificationModule
from torchvision.transforms import Compose


def get_modules(config, shuffle_training: bool = True):
    """
    Returns model and data modules according to Hydra config.
    """
    data_module_cls = hydra.utils.get_class(config.data._target_)
    data_module = data_module_cls(config=config, shuffle=shuffle_training)

    module = ClassificationModule(
        config=config,
        num_classes=data_module.num_classes,
    )

    if config.model.predefined_preprocessing:
        data_module.train_tsfm = (
            Compose([data_module.train_tsfm, module.preprocess])
            if config.trainer.use_train_augmentations
            else Compose([data_module.val_tsfm, module.preprocess])
        )
        data_module.val_tsfm = Compose([data_module.val_tsfm, module.preprocess])

    data_module.create_datasets()

    return data_module, module


def _clean_config_for_backward_compatibility(config):
    delattr(config.data, "augmentations")
    delattr(config.trainer, "patience_for_early_stop")
    delattr(config.trainer, "num_epochs")
    if config.model.checkpoint_name == "None":
        delattr(config.model, "checkpoint_name")
    # delattr(config.trainer, "metric_to_monitor")
    if not config.trainer.use_focal_loss:
        delattr(config.trainer, "focal_loss_gamma")


def get_filter_for_config(config):
    config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    config_as_flat_dict = dict_as_attribute(config)
    cfg_filter = {}
    for k, v in config_as_flat_dict.items():
        if not k.startswith("experiment"):
            # for backward compatibility
            if isinstance(v, bool):
                if not v:
                    cfg_filter[f"config.{k}"] = {"$ne": True}
                else:
                    cfg_filter[f"config.{k}"] = v
            else:
                cfg_filter[f"config.{k}"] = v
    return cfg_filter


def dict_as_attribute(dict):
    result = {}
    for k, v in dict.items():
        if isinstance(v, Dict):
            converted_v = dict_as_attribute(v)
            for k2, v2 in converted_v.items():
                result[f"{k}.{k2}"] = v2
        else:
            result[k] = v
    return result


def get_run_id_from_config(
    config: OmegaConf,
    allow_multiple_runs=False,
    allow_return_none_if_no_runs=False,
    return_running_jobs=False,
) -> str:
    """
    Searches for all runs matching the evaluation Hydra config and return the run id.
    If several are found it returns the one with the best Val/Accuracy.
    """
    api = wandb.Api()
    all_runs_ordered = list(
        api.runs(path=config.project_name, filters=get_filter_for_config(config))
    )
    if return_running_jobs:
        run_ids = [
            r.id
            for r in all_runs_ordered
            if (r.state == "finished" or r.state == "running")
        ]
    else:
        run_ids = [r.id for r in all_runs_ordered if (r.state == "finished")]
    # hack cause W&B bug that makes that run wrongly appear as fail
    for run_ok in ["d353ahxj", "h7k8tzga"]:
        if run_ok in [r.id for r in all_runs_ordered]:
            run_ids += [run_ok]
    if len(run_ids) == 0:
        if allow_return_none_if_no_runs:
            return None
        raise RuntimeError("No runs found")
    if allow_multiple_runs:
        return run_ids
    if len(run_ids) > 1:
        raise RuntimeError(
            f"More than one run matching the config found: {all_runs_ordered}"
        )
    return run_ids[0]
