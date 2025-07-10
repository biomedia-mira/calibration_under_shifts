"""
This script is to launch to ablation study on the calibration effect of OOD dataset size.
"""

from copy import deepcopy
import hydra
import numpy as np


from classification.load_model_and_config import (
    _clean_config_for_backward_compatibility,
    get_modules,
    get_run_id_from_config,
)
from classification.classification_module import ClassificationModule

import torch
import pandas as pd
import pytorch_lightning as pl
from torchmetrics.functional.classification import (
    multiclass_calibration_error,
    accuracy,
    auroc,
)
from calibration.post_hoc_calibrators import (
    EBSCalibrator,
    IRMCalibrator,
    IROvATSCalibrator,
    TemperatureScaler,
)
from pathlib import Path

from calibration.inference_utils import (
    get_outputs,
    to_tensor_if_necessary,
    compute_brier_score,
)
from default_paths import ROOT


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def run_inference(config):
    print("############ Load model ############")
    skip_if_exists = False
    config2 = deepcopy(config)
    _clean_config_for_backward_compatibility(config2)
    run_id = get_run_id_from_config(config2, allow_return_none_if_no_runs=False)
    output_dir = ROOT / Path(f"outputs/run_{run_id}")
    if skip_if_exists and (output_dir / f"ebs_ablation_metrics_ECE.csv").exists():
        print(f"Already exists at {output_dir}")
        return
    print(f"Will be saved at {output_dir}")
    config.trainer.use_train_augmentations = False
    if hasattr(config.data, "cache"):
        config.data.cache = False

    print("############ Load data modules ###########")
    pl.seed_everything(config.seed)
    data_module, _ = get_modules(config, shuffle_training=False)
    model_module = ClassificationModule.load_from_checkpoint(
        f"{output_dir}/best.ckpt", config=config, strict=False
    )
    model_module.get_all_features = True

    trainer = pl.Trainer(enable_progress_bar=True)

    if (output_dir / "train_outputs.pt").exists():
        train_results = torch.load(output_dir / "train_outputs.pt")
    else:
        train_results = get_outputs(
            model_module, data_module.train_dataloader(), trainer
        )
        torch.save(train_results, output_dir / "train_outputs.pt")

    if (output_dir / "val_outputs.pt").exists():
        val_results = torch.load(output_dir / "val_outputs.pt")
    else:
        val_results = get_outputs(model_module, data_module.val_dataloader(), trainer)
        torch.save(val_results, output_dir / "val_outputs.pt")

    if (output_dir / "test_outputs.pt").exists():
        test_results = torch.load(output_dir / "test_outputs.pt")
    else:
        test_results = {}
        for name, loader in data_module.get_evaluation_ood_dataloaders().items():
            test_results[name] = get_outputs(model_module, loader, trainer)

        if hasattr(data_module, "dataset_test"):
            test_results["id"] = get_outputs(
                model_module, data_module.test_dataloader(), trainer
            )

        torch.save(test_results, output_dir / "test_outputs.pt")

    logit_column = "logits"

    ts_calibrator = TemperatureScaler()
    ts_calibrator.fit(val_results[logit_column], val_results["targets"])

    for ood_prop in [0.005, 0.02, 0.10, 0.50, 1.0]:
        print(f"####### Baseline EBS {ood_prop} ######")
        ood_val_results = get_outputs(
            model_module, data_module.get_irrelevant_ood_loader(ood_prop), trainer
        )
        EBS_calibrator = EBSCalibrator(ts_calibrator.t)
        EBS_calibrator.fit(
            val_results[logit_column],
            val_results["targets"],
            ood_val_results[logit_column],
        )

        logits = np.concatenate(
            (val_results[logit_column], ood_val_results[logit_column])
        )
        labels = np.concatenate(
            (
                val_results["targets"],
                np.ones((ood_val_results[logit_column].shape[0])) * -1,
            )
        )

        ts_calibrator_with_ood = TemperatureScaler()
        ts_calibrator_with_ood.fit(logits, labels)

        irm_calibration_with_ood = IRMCalibrator()
        irm_calibration_with_ood.fit(logits, labels)

        irovats_with_ood = IROvATSCalibrator(ts_calibrator_with_ood.t)
        irovats_with_ood.fit(logits, labels)
        for _, outputs in test_results.items():
            outputs[f"calib_ebs_{ood_prop}"] = EBS_calibrator.calibrate(
                outputs[logit_column]
            )
            outputs[f"calib_ts_{ood_prop}"] = ts_calibrator_with_ood.calibrate(
                outputs[logit_column]
            )
            outputs[f"calib_irm_{ood_prop}"] = irm_calibration_with_ood.calibrate(
                outputs[logit_column]
            )
            outputs[f"calib_irovats_{ood_prop}"] = irovats_with_ood.calibrate(
                outputs[logit_column]
            )

    print("#### Compute all metrics for all columns")
    probabilities_columns = [
        x for x in list(test_results.values())[0].keys() if x.startswith("calib")
    ] + ["probas"]

    metrics_list = [
        "ECE",
        "Accuracy",
        "ROCAUC",
        "AvgConfidence",
        "AE_Acc",
        "Brier",
    ]
    metrics = {}
    for m in metrics_list:
        metrics[m] = {n: {} for n in test_results.keys()}
    for name, outputs in test_results.items():
        for c in probabilities_columns:
            outputs[c] = to_tensor_if_necessary(outputs[c])
            metrics["ECE"][name][c] = multiclass_calibration_error(
                outputs[c], outputs["targets"], num_classes=data_module.num_classes
            ).item()
            metrics["Accuracy"][name][c] = accuracy(
                outputs[c],
                outputs["targets"],
                num_classes=data_module.num_classes,
                task="multiclass",
            ).item()
            metrics["ROCAUC"][name][c] = auroc(
                outputs[c],
                outputs["targets"],
                num_classes=data_module.num_classes,
                task="multiclass",
            ).item()
            try:
                confidence = np.max(outputs[c].numpy(), axis=1)
            except TypeError:
                print(confidence)
                print(type(confidence))
                raise TypeError
            print(confidence.shape)
            metrics["AvgConfidence"][name][c] = float(confidence.mean())
            metrics["AE_Acc"][name][c] = float(
                np.abs(metrics["Accuracy"][name][c] - confidence.mean())
            )
            metrics["Brier"][name][c] = compute_brier_score(
                outputs[c], outputs["targets"]
            )

    print("####### Save all metrics #########")
    for k in metrics_list:
        df = pd.DataFrame(metrics[k]).transpose()
        df.to_csv(output_dir / f"ebs_ablation_metrics_{k}.csv")
        print(df)


if __name__ == "__main__":
    """
    Script to run one particular configuration.
    """
    run_inference()
