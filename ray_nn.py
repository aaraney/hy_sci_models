#!/usr/bin/env python3

from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from ray.tune.session import report

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import dataset

from hy_sci_models import models, data


class SingleLayerNet(nn.Module):
    def __init__(
        self, input_dims: int, hidden_layer_dims: int = 12, n_hidden_layers: int = 1
    ) -> None:
        super(SingleLayerNet, self).__init__()

        self.linear_nn = nn.Sequential(
            nn.Linear(input_dims, hidden_layer_dims),
            # nn.BatchNorm1d(hidden_layer_dims),
            # nn.Dropout(0.3),  # Dropout function with 30% prob
            # nn.ReLU(),
            # nn.Linear(hidden_layer_dims, hidden_layer_dims),
            # # nn.BatchNorm1d(hidden_layer_dims),
            # nn.Dropout(0.2),  # Dropout function with 20% prob
            # nn.ReLU(),
            # nn.Linear(hidden_layer_dims, hidden_layer_dims),
            # nn.Dropout(0.2),  # Dropout function with 20% prob
            # nn.ReLU(),
            # nn.Linear(hidden_layer_dims, hidden_layer_dims),
            # nn.Dropout(0.2),  # Dropout function with 20% prob
            # nn.ReLU(),
            # nn.Linear(hidden_layer_dims, hidden_layer_dims),
            # nn.Linear(hidden_layer_dims, 1),
        )

        for i in range(n_hidden_layers):
            self.add_layer(hidden_layer_dims)

        self.linear_nn.add_module(name="output", module=nn.Linear(hidden_layer_dims, 1))

    def forward(self, x):
        return self.linear_nn(x)

    def add_layer(self, hidden_layer_dims: int):
        length_of_modules = len(self.linear_nn._modules.keys())
        module_1_name = str(length_of_modules + 1)
        module_2_name = str(length_of_modules + 2)
        module_3_name = str(length_of_modules + 3)

        # layers to add to network
        self.linear_nn.add_module(name=module_1_name, module=nn.Dropout(0.2))
        self.linear_nn.add_module(name=module_2_name, module=nn.ReLU())
        self.linear_nn.add_module(
            name=module_3_name, module=nn.Linear(hidden_layer_dims, hidden_layer_dims)
        )


def loader(
    labels="mean_depth_va",
    # labels="mean_depth_va_median",
    # labels="mean_depth_va_mean",
    features=[
        "order",
        "area",
        "Sin",
        "Slp",
        "Elev",
        "K",
        "P",
        "AI",
        "LAI",
        "SND",
        "CLY",
        "SLT",
        "Urb",
        # "WTD",
        # "HW",
        "DOR",
        # "stream_wdth_va_median",
        # "stream_wdth_va_mean",
        "stream_wdth_va",
        # "width_m",
        "QMEAN",
    ],
    dataset_path="/media/austinraney/backup_plus/thesis/data/joined_data_subset.p",
    # dataset_path="/media/austinraney/backup_plus/thesis/data/mean_and_median_depth_and_width.p",
    train_size: float = 0.85,
    seed: int = 42,
    **kwargs,
):
    dataset_path = Path(dataset_path).resolve()
    feature_data, label_data = data.setup(dataset_path, features, labels)

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
    # Initialize dataset objects
    train_dataset = data.NNDataset(X_train, y_train)
    val_dataset = data.NNDataset(X_val, y_val)
    test_dataset = data.NNDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset


def train(
    config: dict,
    output_dir: str = None,
) -> Tuple:
    """
    Trains and validates input model.

    Returns:
        model, training loss: List, validation loss: List
    """
    net_neurons = config["net_neurons"]
    epochs = config["epochs"]
    learning_rate = config["lr"]
    training_batchsize = config["batch_size"]
    seed = config["seed"]
    n_hidden_layers = config["n_hidden_layers"]

    # Set numpy and torch seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, val_dataset, _ = loader(seed=seed)

    # Initialize dataset objects
    # train_dataset = data.NNDataset(X_train, y_train)
    # val_dataset = data.NNDataset(X_val, y_val)
    # test_dataset = data.NNDataset(X_test, y_test)

    # Initialize dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_batchsize,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    N_FEATURES = train_dataset.X_data.shape[1]

    # instantiate model
    model = SingleLayerNet(N_FEATURES, net_neurons, n_hidden_layers)

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

    for e in range(1, epochs + 1):

        train_epoch_loss = 0
        validation_epoch_loss = 0
        validation_list = []
        validation_prediction_list = []

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

        model.eval()
        ### VALIDATE MODEL ###
        with torch.no_grad():

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

                validation_list.extend(y_validation_batch.view(-1, 1).cpu().numpy())
                validation_prediction_list.extend(y_val_prediction.cpu().numpy())

        train_epoch_loss = train_epoch_loss / len(train_loader)
        validation_epoch_loss = validation_epoch_loss / len(val_loader)
        validation_list = np.array(validation_list)
        validation_prediction_list = np.array(validation_prediction_list)

        # error = validation_prediction_list - validation_list
        error = validation_list - validation_prediction_list
        rmse = np.sqrt(mean_squared_error(validation_list, validation_prediction_list))
        r2 = r2_score(validation_list, validation_prediction_list)
        std = np.std(error)
        pbias = error.sum() / validation_list.sum()

        # rmse = np.sqrt(
        #     mean_squared_error(
        #         (np.e ** np.array(validation_list)) + 1,
        #         (np.e ** np.array(validation_prediction_list)) + 1,
        #     )
        # )
        # r2 = r2_score(
        #     (np.e ** np.array(validation_list)) + 1,
        #     (np.e ** np.array(validation_prediction_list)) + 1,
        # )

        tune.report(
            loss=validation_epoch_loss,
            training_loss=train_epoch_loss,
            accuracy=rmse,
            r2=r2,
            std=std,
            pbias=pbias,
        )
        # Append loss values to respective lists
        # training_loss_list.append(train_epoch_loss)
        # validation_loss_list.append(validation_epoch_loss)
    print("finished training")


def main():
    SEED = 1024
    NUM_SAMPLES = 20000
    EPOCHS = 60
    # Set numpy and torch seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = {
        "net_neurons": tune.sample_from(lambda _: np.random.randint(1, 20) * 10),
        "epochs": tune.choice([EPOCHS]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        ),
        # "batch_size": tune.choice([5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60]),
        # "batch_size": tune.choice([10, 20, 30, 40, 50, 60]),
        # "batch_size": tune.choice([10, 20, 30, 40, 50, 60]),
        # Include seed for reproducibility
        "seed": SEED,
        # "n_hidden_layers": tune.choice([1]),
        "n_hidden_layers": tune.choice(list(range(2, 12))),
    }

    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=EPOCHS, grace_period=4, reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=[
            "loss",
            "training_loss",
            "accuracy",
            "r2",
            "std",
            "pbias",
            "training_iteration",
        ],
        # report to screen every 10 minutes
        # max_report_frequency=600
        max_report_frequency=60,
    )

    result = tune.run(
        train,
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=NUM_SAMPLES,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
    )

    return result
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )


if __name__ == "__main__":
    result = main()