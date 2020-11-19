#!/usr/bin/env python3

import configparser
import json
import sys
from pathlib import Path

# local imports
from . import workflows
from . import data

SUPPORTED_MODELS = {"nn": workflows.nn, "ols": workflows.ols}

# Major keys are required
MAJOR_KEYS = {
    "batch_size": int,
    "epochs": int,
    "learning_rate": float,
    "dataset_path": str,
    "features": json.loads,
    "labels": str,
    "model": SUPPORTED_MODELS,
}

# Minor keys are optional
MINOR_KEYS = {"n_hidden_layers": int, "train_size": float, "seed": int}


def _handle_config(cfg: configparser.ConfigParser):
    config_dict = cfg["DEFAULT"]

    major_keys_set = set(MAJOR_KEYS)
    minor_keys_set = set(MINOR_KEYS)

    # Check for missing required keys in config
    if not major_keys_set.issubset(config_dict):
        error_message = f"Missing key(s): {major_keys_set.difference(config_dict)}"
        raise KeyError(error_message)

    data.SEED = config_dict.getint("seed")

    model = SUPPORTED_MODELS.get(config_dict.pop("model").lower())
    MAJOR_KEYS.pop("model")

    if model is None:
        error_message = f"Selected model, {model}, does not exist.\nPlease choose from: {list(SUPPORTED_MODELS.keys())}"
        raise KeyError(error_message)

    # Lower case the keys and cast each item to the correct type
    kwargs = {k: v(config_dict[k]) for (k, v) in MAJOR_KEYS.items()}

    # Handle minor keys
    minor_keys_present = {
        k: MINOR_KEYS[k](config_dict[k])
        for k in minor_keys_set.intersection(config_dict)
    }

    kwargs.update(minor_keys_present)
    return model, kwargs


def model_factory(model, kwargs):

    if model == workflows.nn:
        (
            _model,
            training_loss,
            validation_loss,
            train_loader,
            val_loader,
            test_loader,
        ) = model(**kwargs)

        results = {
            "model": _model,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }

        return results

    elif model == workflows.ols:
        _model, X_train, X_test, y_train, y_test = model(**kwargs)
        results = {
            "model": _model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        return results


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

    # (
    #     model,
    #     training_loss,
    #     validation_loss,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    # ) = workflows.nn(**kwargs)
    # # y_list, y_hat_list = nn.test(model, test_loader)

    # return model, training_loss, validation_loss, train_loader, val_loader, test_loader