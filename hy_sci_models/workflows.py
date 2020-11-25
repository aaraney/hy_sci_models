#!/usr/bin/env python3

from sklearn.model_selection import train_test_split

# local imports
from . import data
from . import models


def nn(
    dataset_path,
    features,
    labels,
    epochs,
    batch_size,
    learning_rate,
    n_hidden_layers: int = 12,
    train_size: float = 0.8,
    seed: int = None,
    **kwargs
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
    ) = data.split_dataset_into_train_test_val(
        feature_data, label_data, train_size, seed=seed
    )

    X_train, X_val, X_test, y_train, y_val, y_test = data.fit_and_transform(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    train_loader, val_loader, test_loader = data.generate_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )

    model = models.nn.SingleLayerNet(N_INPUT_FEATURES, n_hidden_layers)

    # Train model
    model, training_loss, validation_loss = models.nn.train(
        model, train_loader, val_loader, epochs, learning_rate
    )

    return model, training_loss, validation_loss, train_loader, val_loader, test_loader


def ols(
    dataset_path, features, labels, train_size: float = 0.8, seed: int = None, **kwargs
):
    feature_data, label_data = data.setup(dataset_path, features, labels)

    if train_size >= 1:
        error_message = "training size cannot greater than or equal to 1"
        raise ValueError(error_message)

    test_size = 1 - train_size

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, label_data, test_size=test_size, random_state=seed
    )

    model = models.ols.OLS(y_train, X_train)
    model = models.ols.train(model)

    return model, X_train, x_test, y_train, y_test
