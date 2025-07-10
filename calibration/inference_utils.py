import torch
import numpy as np


def _concatenate_results(results):
    cat_results = {}
    for k in results[0].keys():
        if isinstance(results[0][k], torch.Tensor):
            cat_results[k] = torch.cat([result_i[k] for result_i in results])
        else:
            # if list of features we want to keep the list structure.
            cat_results[k] = [
                torch.cat([result_i[k][i] for result_i in results])
                for i in range(len(results[0][k]))
            ]
    return cat_results


def compute_brier_score(probas, targets):
    one_hot_labels = torch.zeros_like(probas)
    for k in range(probas.shape[1]):
        one_hot_labels[:, k] = (targets == k).int()
    score = torch.pow((one_hot_labels - probas), 2)
    brier = score.sum(dim=1).mean().item()
    return brier


def get_outputs(
    model_module, loader, trainer, n_output_passes=1, concatenate_multiple_passes=False
):
    results = trainer.predict(model_module, loader)
    list_results = []
    for _ in range(n_output_passes):
        list_results.append(_concatenate_results(results))
    if n_output_passes == 1:
        return list_results[0]
    if concatenate_multiple_passes:
        return _concatenate_results(list_results)
    return list_results


def to_tensor_if_necessary(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array
    if isinstance(tensor_or_array, np.ndarray):
        return torch.tensor(tensor_or_array)
    raise ValueError
