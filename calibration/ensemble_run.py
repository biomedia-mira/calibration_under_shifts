"""
This script launches ensemble inference, including running pre- and post-
ensembling calibration.
"""

from calibration.post_hoc_calibrators import EBSCalibrator, TemperatureScaler

from copy import deepcopy

from collections import defaultdict
import itertools
from pathlib import Path
from hydra import compose, initialize
import numpy as np
import torch
from calibration.inference import add_baseline_calibration_results
from calibration.inference_utils import get_outputs, to_tensor_if_necessary
from classification.classification_module import ClassificationModule
from classification.load_model_and_config import (
    get_run_id_from_config,
    _clean_config_for_backward_compatibility,
    get_modules,
)
import pytorch_lightning as pl
from torchmetrics.functional.classification import (
    multiclass_calibration_error,
    accuracy,
    auroc,
)

from calibration.inference_utils import (
    get_outputs,
    to_tensor_if_necessary,
    compute_brier_score,
)

import pandas as pd
from default_paths import ROOT


foundation_model_base = ["dinov2", "mae", "siglip", "clip"]
experiment_model_map = {
    "base_density": foundation_model_base + ["biomedclip"],
    "base_camelyon": foundation_model_base + ["biomedclip"],
    "base_retina": foundation_model_base + ["biomedclip"],
    "base_living17": foundation_model_base,
    "base_entity30": foundation_model_base,
    "base_domainnet": foundation_model_base,
    "base_icam": foundation_model_base,
    "base_chexpert": foundation_model_base + ["biomedclip"],
}

evaluate_foundation_models = True

all_experiments = [
    "base_density",
    "base_camelyon",
    "base_retina",
    "base_living17",
    "base_entity30",
    "base_domainnet",
    "base_icam",
    "base_chexpert",
]


for experiment in all_experiments:
    print(f"\n\n ################# {experiment.upper()} #################### \n\n")
    for ls, er in [(0, 0), (0.05, 0.1)]:
        if evaluate_foundation_models:
            model_list = experiment_model_map[experiment]
            configs_to_evaluate = [
                [
                    f"experiment={experiment}",
                    f"model={model}",
                    f"model.freeze_encoder=False",
                    "trainer.lr=1e-5",
                    "data.cache=False",
                ]
                for model in model_list
            ]
        else:
            model_names = [
                "resnet18",
                "resnet50",
                "mobilenetv2_100",
                "vit_base_patch16_224",
                "efficientnet_b0",
                "convnext_tiny",
            ]
            configs_to_evaluate = [
                [
                    f"experiment={experiment}",
                    f"model.encoder_name={model}",
                    f"model.pretrained=False",
                ]
                for model in model_names
            ]
        run_ids = []

        trainer = pl.Trainer(enable_progress_bar=True)
        ensemble_output_dir = ROOT / Path(
            f"outputs/ensembling_results/{experiment}/{float(ls):.2f}_{float(er):.2f}_{evaluate_foundation_models}"
        )
        ensemble_output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}
        all_test_results = {}
        all_val_results = {}
        all_ood_val_results = {}

        individual_run_result_df = defaultdict(list)
        with initialize(version_base=None, config_path="../configs"):
            for config_str in configs_to_evaluate:
                config = compose(
                    config_name="config.yaml",
                    overrides=config_str
                    + [
                        f"trainer.label_smoothing={ls}",
                        f"trainer.entropy_regularisation={er}",
                    ],
                )
                config2 = deepcopy(config)
                delattr(config2.trainer, "lr")
                delattr(config2.data, "cache")
                _clean_config_for_backward_compatibility(config2)
                run_id = get_run_id_from_config(
                    config2,
                    allow_multiple_runs=False,
                    allow_return_none_if_no_runs=True,
                )
                if run_id is not None:
                    run_ids.append(run_id)
                pl.seed_everything(config.seed)
                ## Step 1: get the individual results
                output_dir = ROOT / Path(f"outputs/run_{run_id}")
                data_module, _ = get_modules(config, shuffle_training=False)
                model_module = ClassificationModule.load_from_checkpoint(
                    f"{output_dir}/best.ckpt", config=config, strict=False
                )
                model_module.get_all_features = True

                ood_val_results = get_outputs(
                    model_module, data_module.get_irrelevant_ood_loader(0.1), trainer
                )
                train_results = torch.load(output_dir / "train_outputs.pt")
                val_results = torch.load(output_dir / "val_outputs.pt")
                # val_results = get_outputs(model_module, data_module.val_dataloader(), trainer)
                test_results = torch.load(output_dir / "test_outputs.pt")

                print("####### Baseline post-hoc calibrators ######")
                add_baseline_calibration_results(
                    val_results, ood_val_results, test_results, data_module.num_classes
                )

                all_test_results[run_id] = deepcopy(test_results)
                all_val_results[run_id] = deepcopy(val_results)
                all_ood_val_results[run_id] = deepcopy(ood_val_results)

                test_dataset_names = test_results.keys()
                metrics = {}
                metrics_list = ["ECE", "Accuracy", "Brier"]
                probabilities_columns = [
                    x
                    for x in list(test_results.values())[0].keys()
                    if x.startswith("calib")
                ] + ["probas"]
                for m in metrics_list:
                    metrics[m] = {n: {} for n in test_dataset_names}
                for name, outputs in test_results.items():
                    for c in probabilities_columns:
                        outputs[c] = to_tensor_if_necessary(outputs[c])
                        metrics["ECE"][name][c] = multiclass_calibration_error(
                            outputs[c],
                            outputs["targets"],
                            num_classes=data_module.num_classes,
                        ).item()
                        metrics["Accuracy"][name][c] = accuracy(
                            outputs[c],
                            outputs["targets"],
                            num_classes=data_module.num_classes,
                            task="multiclass",
                        ).item()
                        metrics["Brier"][name][c] = compute_brier_score(
                            outputs[c], outputs["targets"]
                        )

                print("####### Save all metrics #########")
                for k in metrics_list:
                    df = pd.DataFrame(metrics[k]).transpose()
                    df["run"] = run_id
                    individual_run_result_df[k].append(df)

        for k in metrics_list:
            df = pd.concat(individual_run_result_df[k])
            df.to_csv(ensemble_output_dir / f"individual_run_metrics_{k}.csv")

        ## Step 2: ensembling
        combinations = list(itertools.combinations(run_ids, 3))
        print(combinations)

        probabilities_columns = [
            x for x in list(test_results.values())[0].keys() if x.startswith("calib")
        ] + ["probas"]
        test_dataset_names = test_results.keys()
        metrics_list = [
            "ECE",
            "Accuracy",
            "ROCAUC",
            "AvgConfidence",
            "AE_Acc",
            "Brier",
        ]

        all_ensemble_df = defaultdict(list)
        combinations_to_test = (
            [
                combinations[k]
                for k in np.random.choice(
                    np.arange(len(combinations)), 5, replace=False
                )
            ]
            if not evaluate_foundation_models
            else combinations
        )

        for ensemble_members in combinations_to_test:
            print(ensemble_members)

            ensemble_test_result = {}
            ensemble_val = {}
            ensemble_ood_val = {}
            # Way 1: simply ensemble all the various calibrated members
            for name in test_dataset_names:
                ensemble_test_result[name] = {
                    "targets": all_test_results[ensemble_members[0]][name]["targets"]
                }
                assert torch.all(
                    all_test_results[ensemble_members[0]][name]["targets"]
                    == all_test_results[ensemble_members[1]][name]["targets"]
                )
                for p in probabilities_columns:
                    ensemble_test_result[name][p] = np.stack(
                        [all_test_results[r][name][p] for r in ensemble_members]
                    ).mean(0)

                ensemble_test_result[name]["pseudo_logits_probas"] = np.log(
                    np.clip(
                        ensemble_test_result[name]["probas"],
                        a_min=1e-12,
                        a_max=1 - 1e-12,
                    )
                )
                ensemble_test_result[name]["pseudo_logits_probas_ebs"] = np.log(
                    np.clip(
                        ensemble_test_result[name]["calib_ebs"],
                        a_min=1e-12,
                        a_max=1 - 1e-12,
                    )
                )
                ensemble_test_result[name]["pseudo_logits_probas_ts"] = np.log(
                    np.clip(
                        ensemble_test_result[name]["calib_ts"],
                        a_min=1e-12,
                        a_max=1 - 1e-12,
                    )
                )
            for p in ["probas", "calib_ebs", "calib_ts"]:
                ensemble_val[p] = np.stack(
                    [all_val_results[r][p] for r in ensemble_members]
                ).mean(0)
            assert torch.all(
                all_val_results[ensemble_members[0]]["targets"]
                == all_val_results[ensemble_members[1]]["targets"]
            )
            # Way 2: post-ensembling calibration
            ensemble_val["pseudo_logits_probas"] = np.log(
                np.clip(ensemble_val["probas"], a_min=1e-12, a_max=1 - 1e-12)
            )
            ensemble_val["pseudo_logits_probas_ebs"] = np.log(
                np.clip(ensemble_val["calib_ebs"], a_min=1e-12, a_max=1 - 1e-12)
            )
            ensemble_val["pseudo_logits_probas_ts"] = np.log(
                np.clip(ensemble_val["calib_ts"], a_min=1e-12, a_max=1 - 1e-12)
            )
            ensemble_ood_val["pseudo_logits_probas"] = np.log(
                np.clip(
                    np.stack(
                        [all_ood_val_results[r]["probas"] for r in ensemble_members]
                    ).mean(0),
                    a_min=1e-12,
                    a_max=1 - 1e-12,
                )
            )
            ensemble_ood_val["pseudo_logits_probas_ebs"] = np.log(
                np.clip(
                    np.stack(
                        [all_ood_val_results[r]["calib_ebs"] for r in ensemble_members]
                    ).mean(0),
                    a_min=1e-12,
                    a_max=1 - 1e-12,
                )
            )
            ensemble_ood_val["pseudo_logits_probas_ts"] = np.log(
                np.clip(
                    np.stack(
                        [all_ood_val_results[r]["calib_ts"] for r in ensemble_members]
                    ).mean(0),
                    a_min=1e-12,
                    a_max=1 - 1e-12,
                )
            )
            print(
                ensemble_val["pseudo_logits_probas"].min(),
                ensemble_val["pseudo_logits_probas"].max(),
            )

            print("##### Post ensembling calibration #####")
            logit_name = "pseudo_logits_probas"
            ts_calibrator = TemperatureScaler()
            ts_calibrator.fit(ensemble_val[logit_name], val_results["targets"])

            EBS_calibrator = EBSCalibrator(ts_calibrator.t)
            EBS_calibrator.fit(
                ensemble_val[logit_name],
                val_results["targets"],
                ensemble_ood_val[logit_name],
            )

            logits = np.concatenate(
                (ensemble_val[logit_name], ensemble_ood_val[logit_name])
            )
            labels = np.concatenate(
                (
                    val_results["targets"],
                    np.ones((ensemble_ood_val[logit_name].shape[0])) * -1,
                )
            )

            ts_calibrator_with_ood = TemperatureScaler()
            ts_calibrator_with_ood.fit(logits, labels)

            for name in test_dataset_names:
                ensemble_test_result[name]["pseudo_logits_probas_ens_ebs"] = (
                    EBS_calibrator.calibrate(ensemble_test_result[name][logit_name])
                )
                ensemble_test_result[name]["pseudo_logits_probas_ens_ts"] = (
                    ts_calibrator.calibrate(ensemble_test_result[name][logit_name])
                )
                ensemble_test_result[name]["pseudo_logits_probas_ens_ts_ood"] = (
                    ts_calibrator_with_ood.calibrate(
                        ensemble_test_result[name][logit_name]
                    )
                )
                ensemble_test_result[name]["pseudo_logits_probas_ens"] = torch.softmax(
                    torch.tensor(ensemble_test_result[name][logit_name]), 1
                )

            metrics = {}
            for m in metrics_list:
                metrics[m] = {n: {} for n in test_dataset_names}
            for name, outputs in ensemble_test_result.items():
                for c in probabilities_columns + [
                    "pseudo_logits_probas_ens_ts",
                    "pseudo_logits_probas_ens_ebs",
                    "pseudo_logits_probas_ens_ts_ood",
                    "pseudo_logits_probas_ens",
                ]:
                    outputs[c] = to_tensor_if_necessary(outputs[c])
                    metrics["ECE"][name][c] = multiclass_calibration_error(
                        outputs[c],
                        outputs["targets"],
                        num_classes=data_module.num_classes,
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
                save_dir = (
                    ensemble_output_dir
                    / f"ensemble_{ensemble_members[0]}_{ensemble_members[1]}_{ensemble_members[2]}"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_dir / f"metrics_{k}.csv")
                all_ensemble_df[k].append(df)
                print(
                    k,
                    df[
                        [
                            "pseudo_logits_probas_ens_ts",
                            "pseudo_logits_probas_ens_ts_ood",
                            "pseudo_logits_probas_ens",
                            "probas",
                        ]
                    ],
                )

        for k in metrics_list:
            df = pd.concat(all_ensemble_df[k])
            df.to_csv(ensemble_output_dir / f"all_ensemble_metrics_{k}.csv")
