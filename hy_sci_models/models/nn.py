#!/usr/bin/env python3

import logging
from typing import Tuple
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass

# local import
from .abstractmodeloutput import AbstractModelOutput

logging.basicConfig(
    level="INFO",
    # format="%(asctime)s %(process)d %(levelname)-8s: %(name)s %(funcName)-15s %(message)s",
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class SingleLayerNet(nn.Module):
    def __init__(self, input_dims: int, hidden_layer_dims: int = 12) -> None:
        super(SingleLayerNet, self).__init__()

        self.linear_nn = nn.Sequential(
            nn.Linear(input_dims, hidden_layer_dims),
            # nn.BatchNorm1d(hidden_layer_dims),
            # nn.Dropout(0.2),  # Dropout function with 20% prob
            nn.ReLU(),
            # nn.Linear(hidden_layer_dims, 8),
            # nn.ReLU(),
            nn.Linear(hidden_layer_dims, 1),
        )

    def forward(self, x):
        return self.linear_nn(x)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
) -> Tuple:
    """
    Trains and validates input model.

    Returns:
        model, training loss: List, validation loss: List
    """
    log = logging.getLogger()

    training_loss_list = []
    validation_loss_list = []

    # Retrieve device information
    device = torch.device("cpu")

    # Define NN model
    # model = SingleLayerNet(N_INPUT_FEATURES)
    model.to(device)

    # Initialize criterion function
    criterion = nn.MSELoss()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ########### Training ###########

    log.info("Beginning Training")

    for e in range(1, epochs + 1):

        train_epoch_loss = 0
        validation_epoch_loss = 0

        ### TRAIN MODEL ###
        # Put model in training mode
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            # Send batches to device
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(
                device
            )

            # zero out gradients
            optimizer.zero_grad()

            y_train_prediction = model(X_train_batch)

            # This might not be necessary the unsqueeze bit
            # training_loss = criterion(y_train_prediction, y_train_batch.unsqueeze(1))
            training_loss = criterion(y_train_prediction, y_train_batch.view(-1, 1))

            training_loss.backward()
            optimizer.step()

            train_epoch_loss += training_loss.item()

        ### VALIDATE MODEL ###
        with torch.no_grad():
            model.eval()

            # Load and send validation to model
            for X_validation_batch, y_validation_batch in val_loader:
                X_validation_batch, y_validation_batch = (
                    X_validation_batch.to(device),
                    y_validation_batch.to(device),
                )

                y_val_prediction = model(X_validation_batch)

                # Quantify loss
                validation_loss = criterion(
                    y_val_prediction, y_validation_batch.view(-1, 1)
                )

                validation_epoch_loss += validation_loss
                # y_pred_list.append(y_val_pred.cpu().numpy())

        train_epoch_loss = train_epoch_loss / len(train_loader)
        validation_epoch_loss = validation_epoch_loss / len(val_loader)

        log.info(f"Epoch: {e}")
        log.info(f"Training loss: {train_epoch_loss}")
        log.info(f"Validation loss: {validation_epoch_loss}")

        # Append loss values to respective lists
        training_loss_list.append(train_epoch_loss)
        validation_loss_list.append(validation_epoch_loss)

    return model, training_loss_list, validation_loss_list


def test(model: nn.Module, test_loader: DataLoader) -> Tuple[np.array]:
    """
    Test model using passed testing data.

    Return:
        y list: np.array, y prediction list: np.array
    """
    # Retrieve device information
    device = torch.device("cpu")

    y_list = []
    y_hat_list = []

    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_test_prediction = model(X_batch)

            y_list.append(y_batch.cpu().numpy())
            y_hat_list.append(y_test_prediction.cpu().numpy())

    y_list = np.concatenate(y_list).squeeze()
    y_hat_list = np.concatenate(y_hat_list).squeeze()

    # NOTE: This is temporary for how the data is preprocessed
    y_list = (np.e ** y_list) - 1
    y_hat_list = (np.e ** y_hat_list) - 1

    return y_list, y_hat_list


@dataclass
class NNModelOutput(AbstractModelOutput):
    training_loss: np.array
    validation_loss: np.array

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader