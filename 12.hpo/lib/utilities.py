"""Helper functions for data conversion and file handling."""

import collections
import os
import pickle
import random
from pathlib import Path
from typing import Dict, Tuple

import neps
import numpy as np
import torch
import torch.nn as nn
from neps.plot.read_results import process_seed
from torch.utils.data import DataLoader


def save_result(filename: str, obj: object) -> None:
    """Save object to disk as pickle file.

    Args:
        filename: Name of file in ./results directory to write object to.
        obj: The object to write to file.

    """
    # make sure save directory exists
    save_path = Path("results")
    os.makedirs(save_path, exist_ok=True)

    # save the python objects as bytes
    with (save_path / f"{filename}.pkl").open("wb") as fh:
        pickle.dump(obj, fh)


def load_result(
    filename: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load object from pickled file.

    Args:
        filename: Name of file in ./results directory to load.

    """
    with (Path("results") / f"{filename}.pkl").open("rb") as fh:
        return pickle.load(fh)


def evaluate_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    """Evaluate the performance of a given model.

    Args:
        model: PyTorch model
        data_loader : Validation data structe

    Returns:
        accuracy

    Note:

    """
    set_seed(111)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            output = model(x)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    accuracy = correct / len(data_loader.sampler)
    return accuracy


def get_results(results) -> dict:
    """Get the results in an easy to read dictionary

    Args:
      results: Path of the results folder.

    Returns:
        Dictionary containing the final losses, hyperparameter configurations, validation errors
        across epochs and incumbents
    """

    summary = neps.get_summary_dict(results)
    losses_and_configs = {
        "Losses": [],
        "Config": [],
        "Val_errors": [],
        "Incumbents": [],
        "Epochs": [],
    }
    config_results, _ = neps.status(results)
    config_results = dict(sorted(config_results.items()))
    lcs = {str(i): [] for i in range(summary["num_evaluated_configs"])}
    for config, result in config_results.items():
        losses_and_configs["Losses"].append(result.result["loss"])
        losses_and_configs["Config"].append(result.config)
        if "_" in config:
            config_id = config.split("_")[0]
            lcs[config_id].extend(result.result["info_dict"]["val_errors"])
        else:
            losses_and_configs["Val_errors"].append(
                result.result["info_dict"]["val_errors"]
            )
    for config_id, vals in lcs.items():
        if len(vals) != 0:
            losses_and_configs["Val_errors"].append(lcs[config_id])

    incumbent, costs, _ = process_seed(
        path=results,
        seed=None,
        key_to_extract="cost",
        consider_continuations=False,
        n_workers=1,
    )
    losses_and_configs["Incumbents"].extend(incumbent)
    losses_and_configs["Epochs"].extend(
        [np.sum(costs[:i]) for i in range(1, len(costs) + 1)]
    )
    return losses_and_configs


def set_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
