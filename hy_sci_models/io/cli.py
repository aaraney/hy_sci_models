#!/usr/bin/env python3

import configparser
import json
import sys
from pathlib import Path

# local imports
from hy_sci_models import workflows, data

# TODO: this is staring too look like a class heirachy with models being subclasses.
# You should do that instead of this...


OLS_MAJOR = {
    "dataset_path": str,
    "features": json.loads,
    "labels": str,
}

OLS_MINOR = {"train_size": float, "seed": int}

# Major keys are required
NN_MAJOR = {
    "batch_size": int,
    "epochs": int,
    "learning_rate": float,
    "dataset_path": str,
    "features": json.loads,
    "labels": str,
}

# Minor keys are optional
NN_MINOR = {"n_hidden_layers": int, "train_size": float, "seed": int}

SUPPORTED_MODELS = {
    "nn": {"workflow": workflows.nn, "major": NN_MAJOR, "minor": NN_MINOR},
    "ols": {"workflow": workflows.ols, "major": OLS_MAJOR, "minor": OLS_MINOR},
}


def _handle_config(cfg: configparser.ConfigParser):
    config_dict = cfg["DEFAULT"]

    model = SUPPORTED_MODELS.get(config_dict.pop("model").lower())

    if model is None:
        error_message = f"Selected model, {model}, does not exist.\nPlease choose from: {list(SUPPORTED_MODELS.keys())}"
        raise KeyError(error_message)

    workflow = model["workflow"]
    major_keys_set = model["major"]
    minor_keys_set = model["minor"]

    # Check for missing required keys in config
    if not set(major_keys_set).issubset(config_dict):
        error_message = f"Missing key(s): {set(major_keys_set).difference(config_dict)}"
        raise KeyError(error_message)

    # Lower case the keys and cast each item to the correct type
    kwargs = {k: v(config_dict[k]) for (k, v) in major_keys_set.items()}

    # TODO: Add this to abstract model config
    kwargs["seed"] = config_dict.getint("seed", None)
    config_dict.pop("seed")

    # Handle minor keys
    minor_keys_present = {
        k: minor_keys_set[k](config_dict[k])
        for k in set(minor_keys_set).intersection(config_dict)
    }

    kwargs.update(minor_keys_present)
    return workflow, kwargs


def model_factory(model, kwargs):
    return model(**kwargs)


def main() -> None:
    if len(sys.argv) < 1:
        print("Requires config file as command line input")
        exit(0)

    cfg = Path(sys.argv[1])

    if not cfg.is_file():
        error_message = "Input config file does not exist."
        raise FileExistsError(error_message)

    config = configparser.ConfigParser()
    config.read(cfg)
    model, kwargs = _handle_config(config)

    return model_factory(model, kwargs)