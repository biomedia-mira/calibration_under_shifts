"""
This script is to run inference and compute all post-hoc calibration results. Metrics are saved to
the model directory.

Run it with
python calibration/inference.py experiment=base_camelyon

This will throw an error if no corresponding model can be found.
"""

from copy import deepcopy
import hydra
import numpy as np

from calibration.inference_utils import (
    get_outputs,
    to_tensor_if_necessary,
    compute_brier_score,
)

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
    ETSCalibrator,
    IRMCalibrator,
    IROvACalibrator,
    IROvATSCalibrator,
    TemperatureScaler,
)
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path
from calibration.density_aware import DAC
from default_paths import ROOT


def add_baseline_calibration_results(
    val_results,
    ood_val_results,
    test_results,
    num_classes,
    suffix="",
    logit_column_val="logits",
    logit_column_test="logits",
):
    """
    This function takes in val, test and semantic OOD predictions to run all post-hoc calibration
    baselines.
    """
    ts_calibrator = TemperatureScaler()
    ts_calibrator.fit(val_results[logit_column_val], val_results["targets"])

    ets_calibrator = ETSCalibrator(ts_calibrator.t, num_classes)
    ets_calibrator.fit(
        val_results[logit_column_val],
        val_results["targets"],
    )

    irm_calibrator = IRMCalibrator()
    irm_calibrator.fit(val_results[logit_column_val], val_results["targets"])

    irova_calibrator = IROvACalibrator()
    irova_calibrator.fit(val_results[logit_column_val], val_results["targets"])
    irovats_calibrator = IROvATSCalibrator(ts_calibrator.t)
    irovats_calibrator.fit(val_results[logit_column_val], val_results["targets"])

    EBS_calibrator = EBSCalibrator(ts_calibrator.t)
    EBS_calibrator.fit(
        val_results[logit_column_val],
        val_results["targets"],
        ood_val_results[logit_column_val],
    )

    ebs_minus = EBSCalibrator(ts_calibrator.t)
    ebs_minus.fit(
        val_results[logit_column_val],
        val_results["targets"],
        None,
    )

    logits = np.concatenate(
        (val_results[logit_column_val], ood_val_results[logit_column_val])
    )
    labels = np.concatenate(
        (
            val_results["targets"],
            np.ones((ood_val_results[logit_column_val].shape[0])) * -1,
        )
    )

    ts_calibrator_with_ood = TemperatureScaler()
    ts_calibrator_with_ood.fit(logits, labels)

    irm_calibration_with_ood = IRMCalibrator()
    irm_calibration_with_ood.fit(logits, labels)

    irovats_with_ood = IROvATSCalibrator(ts_calibrator_with_ood.t)
    irovats_with_ood.fit(logits, labels)

    for _, outputs in test_results.items():
        outputs[f"calib_ts{suffix}"] = ts_calibrator.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_ets{suffix}"] = ets_calibrator.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_irm{suffix}"] = irm_calibrator.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_irova{suffix}"] = irova_calibrator.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_irovats{suffix}"] = irovats_calibrator.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_ebs{suffix}"] = EBS_calibrator.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_ebs_minus{suffix}"] = ebs_minus.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_ts_with_ood{suffix}"] = ts_calibrator_with_ood.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_irm_with_ood{suffix}"] = irm_calibration_with_ood.calibrate(
            outputs[logit_column_test]
        )
        outputs[f"calib_irovats_with_ood{suffix}"] = irovats_with_ood.calibrate(
            outputs[logit_column_test]
        )

    val_results[f"calib_ts{suffix}"] = ts_calibrator.calibrate(
        val_results[logit_column_test]
    )
    val_results[f"calib_ebs{suffix}"] = EBS_calibrator.calibrate(
        val_results[logit_column_test]
    )
    val_results[f"calib_ts_with_ood{suffix}"] = ts_calibrator_with_ood.calibrate(
        val_results[logit_column_test]
    )
    val_results[f"calib_irovats_with_ood{suffix}"] = irovats_with_ood.calibrate(
        val_results[logit_column_test]
    )

    ood_val_results[f"calib_ts{suffix}"] = ts_calibrator.calibrate(
        ood_val_results[logit_column_test]
    )
    ood_val_results[f"calib_ebs{suffix}"] = EBS_calibrator.calibrate(
        ood_val_results[logit_column_test]
    )
    ood_val_results[f"calib_ts_with_ood{suffix}"] = ts_calibrator_with_ood.calibrate(
        ood_val_results[logit_column_test]
    )
    ood_val_results[f"calib_irovats_with_ood{suffix}"] = irovats_with_ood.calibrate(
        ood_val_results[logit_column_test]
    )


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def run_inference(config):
    print("############ Load model ############")
    skip_if_exists = False
    config2 = deepcopy(config)
    _clean_config_for_backward_compatibility(config2)
    run_id = get_run_id_from_config(config2, allow_return_none_if_no_runs=False)
    output_dir = ROOT / Path(f"outputs/run_{run_id}")
    if skip_if_exists and (output_dir / f"metrics_BalAccuracy.csv").exists():
        if (
            "calib_irovats_with_ood"
            in pd.read_csv(output_dir / f"metrics_ECE.csv").columns
        ):
            print(f"Already exists at {output_dir}")
            return
    fixed_ok = (output_dir / f"metrics_BalAccuracy.csv").exists()
    print(f"Will be saved at {output_dir}")
    config.trainer.use_train_augmentations = False
    if hasattr(config.data, "cache"):
        config.data.cache = False

    print("############ Load data modules ###########")
    pl.seed_everything(config.seed)
    data_module, _ = get_modules(config, shuffle_training=False)
    try:
        print(data_module.dataset_name + "\n")
    except NotImplementedError:
        pass
    model_module = ClassificationModule.load_from_checkpoint(
        f"{output_dir}/best.ckpt", config=config, strict=False
    )
    model_module.get_all_features = True

    trainer = pl.Trainer(enable_progress_bar=True)

    if fixed_ok and (output_dir / "train_outputs.pt").exists():
        train_results = torch.load(output_dir / "train_outputs.pt")
    else:
        train_results = get_outputs(
            model_module, data_module.train_dataloader(), trainer
        )
        torch.save(train_results, output_dir / "train_outputs.pt")

    if fixed_ok and (output_dir / "val_outputs.pt").exists():
        val_results = torch.load(output_dir / "val_outputs.pt")
    else:
        val_results = get_outputs(model_module, data_module.val_dataloader(), trainer)
        torch.save(val_results, output_dir / "val_outputs.pt")

    ood_val_results = get_outputs(
        model_module, data_module.get_irrelevant_ood_loader(0.1), trainer
    )

    if fixed_ok and (output_dir / "test_outputs.pt").exists():
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

    print("####### Baseline post-hoc calibrators ######")
    add_baseline_calibration_results(
        val_results, ood_val_results, test_results, data_module.num_classes
    )

    print("#### Baseline post-hoc calibrators with DAC #####")
    suffix = ""
    dac = DAC(
        output_dir=output_dir,
        number_features=len(train_results["feats"]),
        suffix_filenames=suffix,
    )
    dac.fit_knn_scorers(train_results["feats"])
    dac.optimize(
        val_results["logits"].numpy(),
        val_results["targets"].numpy(),
        val_results["feats"],
    )

    val_results["logits_after_dac"] = dac.calibrate_before_softmax(
        val_results["logits"], val_results["feats"], "val"
    )
    ood_val_results["logits_after_dac"] = dac.calibrate_before_softmax(
        ood_val_results["logits"], ood_val_results["feats"], "ood_val"
    )
    for name, outputs in test_results.items():
        outputs["logits_after_dac"] = dac.calibrate_before_softmax(
            outputs["logits"], outputs["feats"], name
        )

    add_baseline_calibration_results(
        val_results,
        ood_val_results,
        test_results,
        data_module.num_classes,
        suffix="_dac" + suffix,
        logit_column_val=f"logits_after_dac{suffix}",
        logit_column_test=f"logits_after_dac{suffix}",
    )

    print("#### Compute all metrics for all columns")
    probabilities_columns = [
        x for x in list(test_results.values())[0].keys() if x.startswith("calib")
    ] + ["probas"]

    metrics_list = [
        "ECE",
        "Accuracy",
        "BalAccuracy",
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
            metrics["BalAccuracy"][name][c] = balanced_accuracy_score(
                outputs["targets"].numpy(), np.argmax(outputs[c].numpy(), 1)
            )
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
        df.to_csv(output_dir / f"metrics_{k}.csv")


if __name__ == "__main__":
    """
    Script to run one particular configuration.
    """
    run_inference()
