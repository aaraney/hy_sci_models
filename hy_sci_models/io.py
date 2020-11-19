#!/usr/bin/env python3

import configparser
import json
import sys
from pathlib import Path

# local imports
from . import data
from .models import nn

# Major keys are required
MAJOR_KEYS = {
    "batch_size": int,
    "epochs": int,
    "learning_rate": float,
    "dataset_path": str,
    "features": json.loads,
    "labels": str,
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

    # Lower case the keys and cast each item to the correct type
    kwargs = {k: v(config_dict[k]) for (k, v) in MAJOR_KEYS.items()}

    # Handle minor keys
    minor_keys_present = {
        k: MINOR_KEYS[k](config_dict[k])
        for k in minor_keys_set.intersection(config_dict)
    }

    kwargs.update(minor_keys_present)
    return kwargs


def workflow(
    dataset_path,
    features,
    labels,
    epochs,
    batch_size,
    learning_rate,
    n_hidden_layers: int = 12,
    train_size: float = 0.8,
):
    feature_data, label_data = data.setup(dataset_path, features, labels)

    # Number of neurons in input layer
    N_INPUT_FEATURES = feature_data.shape[1]

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = data.split_dataset_into_train_test_val(feature_data, label_data, train_size)

    X_train, X_val, X_test, y_train, y_val, y_test = data.fit_and_transform(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    train_loader, val_loader, test_loader = data.generate_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )

    model = nn.SingleLayerNet(N_INPUT_FEATURES, n_hidden_layers)

    # Train model
    model, training_loss, validation_loss = nn.train(
        model, train_loader, val_loader, epochs, learning_rate
    )

    return model, training_loss, validation_loss

    # y_list, y_hat_list = nn.test(model, test_loader)


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
    kwargs = _handle_config(config)

    model, training_loss, validation_loss = workflow(**kwargs)

    # y_list, y_hat_list = nn.test(model, test_loader)