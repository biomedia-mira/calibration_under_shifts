import sys

from matplotlib import ticker

sys.path.append("/vol/biomedic3/mb121/calibration_exploration/")

from classification.load_model_and_config import (
    get_run_id_from_config,
    _clean_config_for_backward_compatibility,
)
from hydra import initialize, compose
from pathlib import Path
import pandas as pd
import numpy as np
import numpy as np
import matplotlib


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


def get_all_runs(experiment, evaluate="scratch"):
    match evaluate:
        case "scratch":
            evaluate_foundation_model = False
            evaluate_both = False
        case "foundation":
            evaluate_foundation_model = True
            evaluate_both = False
        case "both":
            evaluate_both = True
        case _:
            raise ValueError
    if evaluate_both or evaluate_foundation_model:
        model_list = experiment_model_map[experiment]
        configs_to_evaluate = [
            [
                f"experiment={experiment}",
                f"model={model}",
                f"model.freeze_encoder=False",
                "trainer.lr=1e-5",
            ]
            for model in model_list
        ]
        with initialize(version_base=None, config_path="../configs"):
            run_ids = []
            run_ids_ls = []
            run_ids_er = []
            run_ids_er_ls = []
            run_focal = []
            for config_str in configs_to_evaluate:
                config = compose(
                    config_name="config.yaml",
                    overrides=config_str + ["trainer.label_smoothing=0.00"],
                )
                # delattr(config.trainer, 'lr')
                _clean_config_for_backward_compatibility(config)
                run_id = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id is not None:
                    run_ids.append(run_id)
                config = compose(
                    config_name="config.yaml",
                    overrides=config_str + ["trainer.label_smoothing=0.05"],
                )
                # delattr(config.trainer, 'lr')
                _clean_config_for_backward_compatibility(config)
                run_id_ls = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id_ls is not None:
                    run_ids_ls.append(run_id_ls)

                config = compose(
                    config_name="config.yaml",
                    overrides=config_str + ["trainer.entropy_regularisation=0.1"],
                )
                # delattr(config.trainer, 'lr')
                _clean_config_for_backward_compatibility(config)
                run_id_er = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id_er is not None:
                    run_ids_er.append(run_id_er)

                config = compose(
                    config_name="config.yaml",
                    overrides=config_str
                    + [
                        "trainer.entropy_regularisation=0.1",
                        "trainer.label_smoothing=0.05",
                    ],
                )
                # delattr(config.trainer, 'lr')
                _clean_config_for_backward_compatibility(config)
                run_id_er_ls = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id_er_ls is not None:
                    run_ids_er_ls.append(run_id_er_ls)

                config = compose(
                    config_name="config.yaml",
                    overrides=config_str
                    + ["trainer.use_focal_loss=True", "trainer.focal_loss_gamma=-53"],
                )
                _clean_config_for_backward_compatibility(config)
                run_id = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id is not None:
                    run_focal.append(run_id)
        done = (
            len(run_ids)
            == len(configs_to_evaluate)
            == len(run_ids_ls)
            == len(run_ids_er)
            == len(run_ids_er_ls)
            == len(run_focal)
        )
        inference_done = sum(
            [
                (
                    Path(
                        f"/vol/biomedic3/mb121/calibration_exploration/outputs/run_{r}/metrics_ECE.csv"
                    ).exists()
                    and (
                        "calib_ebs_minus"
                        in pd.read_csv(
                            f"/vol/biomedic3/mb121/calibration_exploration/outputs/run_{r}/metrics_ECE.csv"
                        ).columns
                    )
                )
                for r in (run_ids + run_ids_ls + run_ids_er + run_ids_er_ls + run_focal)
            ]
        )
        total = (
            len(run_ids)
            + len(run_ids_ls)
            + len(run_ids_er)
            + len(run_ids_er_ls)
            + len(run_focal)
        )
        print(
            experiment,
            "DONE" if done else "",
            f"{len(run_ids)}/{len(configs_to_evaluate)}",
            f"{len(run_ids_ls)}/{len(configs_to_evaluate)}",
            f"{len(run_ids_er)}/{len(configs_to_evaluate)}",
            f"{len(run_ids_er_ls)}/{len(configs_to_evaluate)}",
            f"{len(run_focal)}/{len(configs_to_evaluate)}",
            f"metrics: {inference_done}",
            "COMPLETE" if done and total == inference_done else "",
        )
    if evaluate_both or (not evaluate_foundation_model):
        pretrained = False
        model_names = [
            "resnet18",
            "resnet50",
            "mobilenetv2_100",
            "convnext_tiny",
            "vit_base_patch16_224",
            "efficientnet_b0",
        ]  #
        configs_to_evaluate = [
            [
                f"experiment={experiment}",
                f"model.encoder_name={model}",
                f"model.pretrained={pretrained}",
            ]
            for model in model_names
        ]

        with initialize(version_base=None, config_path="../configs"):
            if not evaluate_both:
                run_ids = []
                run_ids_ls = []
                run_ids_er = []
                run_ids_er_ls = []
                run_focal = []
            for config_str in configs_to_evaluate:
                config = compose(
                    config_name="config.yaml",
                    overrides=config_str + ["trainer.label_smoothing=0.00"],
                )
                delattr(config.trainer, "lr")
                _clean_config_for_backward_compatibility(config)
                run_id = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id is not None:
                    run_ids.append(run_id)
                config = compose(
                    config_name="config.yaml",
                    overrides=config_str + ["trainer.label_smoothing=0.05"],
                )
                delattr(config.trainer, "lr")
                _clean_config_for_backward_compatibility(config)
                run_id_ls = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id_ls is not None:
                    run_ids_ls.append(run_id_ls)

                config = compose(
                    config_name="config.yaml",
                    overrides=config_str + ["trainer.entropy_regularisation=0.1"],
                )
                delattr(config.trainer, "lr")
                _clean_config_for_backward_compatibility(config)
                run_id_er = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id_er is not None:
                    run_ids_er.append(run_id_er)

                config = compose(
                    config_name="config.yaml",
                    overrides=config_str
                    + [
                        "trainer.entropy_regularisation=0.1",
                        "trainer.label_smoothing=0.05",
                    ],
                )
                delattr(config.trainer, "lr")
                _clean_config_for_backward_compatibility(config)
                run_id_er_ls = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id_er_ls is not None:
                    run_ids_er_ls.append(run_id_er_ls)

                config = compose(
                    config_name="config.yaml",
                    overrides=config_str
                    + ["trainer.use_focal_loss=True", "trainer.focal_loss_gamma=-53"],
                )
                delattr(config.trainer, "lr")
                _clean_config_for_backward_compatibility(config)
                run_id = get_run_id_from_config(
                    config, allow_multiple_runs=False, allow_return_none_if_no_runs=True
                )
                if run_id is not None:
                    run_focal.append(run_id)
        done = (
            len(configs_to_evaluate)
            == len(run_ids)
            == len(run_ids_ls)
            == len(run_ids_er)
            == len(run_ids_er_ls)
            == len(run_focal)
        )
        total = (
            len(run_ids)
            + len(run_ids_ls)
            + len(run_ids_er)
            + len(run_ids_er_ls)
            + len(run_focal)
        )

        inference_done = sum(
            [
                (
                    Path(
                        f"/vol/biomedic3/mb121/calibration_exploration/outputs/run_{r}/metrics_ECE.csv"
                    ).exists()
                    and (
                        "calib_ebs_minus"
                        in pd.read_csv(
                            f"/vol/biomedic3/mb121/calibration_exploration/outputs/run_{r}/metrics_ECE.csv"
                        ).columns
                    )
                )
                for r in (run_ids + run_ids_ls + run_ids_er + run_ids_er_ls + run_focal)
            ]
        )
        print(
            experiment,
            "DONE" if done else "",
            pretrained,
            len(run_ids),
            len(run_ids_ls),
            len(run_ids_er),
            len(run_ids_er_ls),
            len(run_focal),
            f"metrics ok {inference_done} / {total}",
            "COMPLETE" if done and total == inference_done else "",
        )
    return {
        "CE": run_ids,
        "LS": run_ids_ls,
        "ER": run_ids_er,
        "ER+LS": run_ids_er_ls,
        "Focal": run_focal,
    }


def retrieve_metrics_df(list_run_ids, metric):
    all_df = []
    for run_id in list_run_ids:
        output_dir = Path(
            f"/vol/biomedic3/mb121/calibration_exploration/outputs/run_{run_id}"
        )
        try:
            df = pd.read_csv(output_dir / f"metrics_{metric}.csv")
        except FileNotFoundError:
            print(str(output_dir / f"metrics_{metric}.csv") + " Not found")
            continue
        df.rename(columns={"Unnamed: 0": "domain"}, inplace=True)
        if "brightness_s0" in df.domain.values:
            df["domain"] = df["domain"].apply(
                lambda x: (int(x[-1]) + 1) if x != "id" else "id"
            )
        all_df.append(df)

    # print(len(all_df), len(list_run_ids))
    if len(all_df) == 0:
        return pd.DataFrame()
    return pd.concat(all_df)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap


def my_pretty_plot(experiment, f, ax=None, fmt="{x:.2f}"):
    match experiment:
        case "base_chexpert":
            f.suptitle("$\mathbf{CXR}$")
        case "base_density":
            f.suptitle("$\mathbf{EMBED}$")
        case _:
            f.suptitle("$\mathbf{" + experiment.replace("base_", "").upper() + "}$")

    if ax is not None:
        ax[0].get_legend().remove()
        ax[0].set_ylabel("SHIFTED")
        ax[0].set_xlabel("ID")
        ax[1].set_ylabel("SHIFTED")
        ax[1].set_xlabel("ID")
        ax[0].set_title("ECE")
        ax[1].set_title("Brier")
        ax[1].set_ylabel("")
        if fmt is not None:
            ax[0].xaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
            ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
            ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
            ax[1].yaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
