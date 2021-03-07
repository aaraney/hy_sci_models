#!/usr/bin/env python3

from typing import Iterable, Tuple
from pathlib import Path, PurePath
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
import time

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


# Local imports
from hy_sci_models import models, data


class SingleLayerNet(nn.Module):
    def __init__(
        self, input_dims: int, hidden_layer_dims: int = 12, n_hidden_layers: int = 1
    ) -> None:
        super(SingleLayerNet, self).__init__()

        self.linear_nn = nn.Sequential(
            nn.Linear(input_dims, hidden_layer_dims),
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
    features=[
        "COMID",  # For grouping only
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
    seed: int = 42,
):

    dataset_path = Path(dataset_path).resolve()

    # Read dataset from pickle file
    df = pd.read_pickle(dataset_path)

    # Randomly shuffle rows in the dataset. Use reproducible seed. Drop index.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Change type of COMID to str
    df["COMID"] = df.COMID.apply(str)

    feature_data, label_data = data.setup(df, features, labels)

    return feature_data, label_data


def fit_and_transform(X_train, X_test, y_train, y_test):
    """Returned as X_train, X_test, y_train, y_test"""
    # Scaling input training set
    scaler = MinMaxScaler()

    # Fit and transform training set
    X_train = scaler.fit_transform(X_train)

    # Transform testing dataset using the fit of the training set
    X_test = scaler.transform(X_test)

    # Potentially unnecessary casting?
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, X_test, y_train, y_test


def train(
    config: dict,
    train_dataset,
    test_dataset,
    round_number: int = np.nan,
    # output_dir: str = None,
) -> Tuple:
    """
    Trains and validates input model.

    Returns:
        model, training loss: List, validation loss: List
    """
    try:
        net_neurons = config["net_neurons"]
        epochs = config["epochs"]
        learning_rate = config["lr"]
        training_batchsize = config["batch_size"]
        seed = config["seed"]
        n_hidden_layers = config["n_hidden_layers"]

        # Set numpy and torch seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize dataloader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=training_batchsize,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        N_FEATURES = train_dataset.X_data.shape[1]

        # instantiate model
        model = SingleLayerNet(N_FEATURES, net_neurons, n_hidden_layers)

        # Retrieve device information
        device = torch.device("cpu")

        # Define NN model
        model.to(device)

        # Initialize criterion function
        criterion = nn.MSELoss()

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        ########### Training ###########

        for e in range(1, epochs + 1):
            print(f"Round: {round_number}\nEpoch: {e}")

            ### TRAIN MODEL ###
            # Put model in training mode
            model.train()
            for X_train_batch, y_train_batch in train_loader:
                # Send batches to device
                X_train_batch, y_train_batch = (
                    X_train_batch.to(device),
                    y_train_batch.to(device),
                )

                # zero out gradients
                optimizer.zero_grad()

                y_train_prediction = model(X_train_batch)

                # This might not be necessary the unsqueeze bit
                # training_loss = criterion(y_train_prediction, y_train_batch.unsqueeze(1))
                training_loss = criterion(y_train_prediction, y_train_batch.view(-1, 1))

                training_loss.backward()
                optimizer.step()

        print("FINISHED TRAINING MODEL...\nBEGINNING TESTING")

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

        if not isinstance(y_list, Iterable):
            y_list = [y_list]

        if not isinstance(y_hat_list, Iterable):
            y_hat_list = [y_hat_list]

        df = pd.DataFrame({"y": y_list, "y_hat": y_hat_list})

        df["round_number"] = round_number

        return df

    # Broad except statement for if a model fails
    except:
        return pd.DataFrame(
            {"y": [np.nan], "y_hat": [np.nan], "round_number": round_number}
        )


def train_model(X_train, y_train, X_test, y_test, config, round_number):
    # Fit and transform the data
    X_train, X_test, y_train, y_test = fit_and_transform(
        X_train, X_test, y_train, y_test
    )

    # Initialize dataset objects
    train_dataset = data.NNDataset(X_train, y_train)
    test_dataset = data.NNDataset(X_test, y_test)

    df = train(config, train_dataset, test_dataset, round_number=round_number)

    return df


def get_fold(df: pd.DataFrame, label: str, grouping_feature: str):
    """
    training, testing
    """
    # Get unique groups
    groups = df[grouping_feature].unique()

    for item in groups:
        # Yield fold. Drop grouping feature
        training = df[df[grouping_feature] != item].drop(grouping_feature, axis=1)
        testing = df[df[grouping_feature] == item].drop(grouping_feature, axis=1)

        # train_dataset, test_dataset. dtype: torch.utils.data.Dataset
        training, testing = preprocess(training, testing, label)

        yield training, testing


def preprocess(training: pd.DataFrame, testing: pd.DataFrame, label: str):
    """
    train_dataset, test_dataset
    """
    X_train = training[training.columns.difference([label])]
    y_train = training[label]

    X_test = testing[testing.columns.difference([label])]
    y_test = testing[label]

    # Fit and transform the data
    X_train, X_test, y_train, y_test = fit_and_transform(
        X_train, X_test, y_train, y_test
    )

    # Initialize dataset objects
    train_dataset = data.NNDataset(X_train, y_train)
    test_dataset = data.NNDataset(X_test, y_test)

    return train_dataset, test_dataset


def main():
    KFOLDS = 20

    BATCH_SIZE = 10
    EPOCHS = 60
    LEARNING_RATE = 0.000312
    N_HIDDEN = 3
    NET_NEURONS = 170

    SEED = 1024
    EPOCHS = 60

    N_THREADS = 7
    torch.set_num_threads(N_THREADS)

    # kfold = KFold(KFOLDS)

    # Set numpy and torch seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = {
        "net_neurons": NET_NEURONS,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        # Include seed for reproducibility
        "seed": SEED,
        "n_hidden_layers": N_HIDDEN,
    }

    # Load feature data
    # feature_data, label_data = loader(seed=SEED)

    # # Re-join feature label and feature data into a single df
    # feature_data[label_data.name] = label_data

    # # For readability rename
    # joined_feature_and_label_data = feature_data

    # # List of dataframes containing results
    # dfs = []

    # packaged_training_and_test_data = []
    # round_number = 0
    # i = 0

    # g = get_fold(joined_feature_and_label_data, "mean_depth_va", "COMID")
    # # for i in range(8):
    # for train_dataset, test_dataset in g:
    #     # train_dataset, test_dataset = next(g)
    #     train_kwargs = [
    #         config,
    #         train_dataset,
    #         test_dataset,
    #         i,  # Round number
    #     ]
    #     packaged_training_and_test_data.append(train_kwargs)
    #     i += 1

    # with open("at-a-station-cross-validation.p", "wb") as fp:
    #     pickle.dump(packaged_training_and_test_data, fp)

    with open("at-a-station-cross-validation.p", "rb") as fp:
        packaged_training_and_test_data = pickle.load(fp)

    # len_train_and_test_data = len(packaged_training_and_test_data)

    # First quarter
    # packaged_training_and_test_data = packaged_training_and_test_data[:7]
    # packaged_training_and_test_data = packaged_training_and_test_data[
    #     : len_train_and_test_data // 8
    # ]

    # Second quarter
    # packaged_training_and_test_data = packaged_training_and_test_data[
    #     len_train_and_test_data // 4 : (len_train_and_test_data // 4) * 2
    # ]

    # Third quarter
    # packaged_training_and_test_data = packaged_training_and_test_data[
    #     (len_train_and_test_data // 4) * 2 : (len_train_and_test_data // 4) * 3
    # ]

    # Fourth quarter
    # packaged_training_and_test_data = packaged_training_and_test_data[
    #     (len_train_and_test_data // 4)*3 :
    # ]

    with Pool(processes=8) as pool:
        # for package in packaged_training_and_test_data:
        dfs = []
        c_i = 0
        count = 1
        for i in range(20, len(packaged_training_and_test_data) + 18, 20):
            dfs.append(pool.starmap(train, packaged_training_and_test_data[c_i:i]))

            count += 1

            # Write a checkpoint every 5 iterations of the inner for loop
            checkpoint_file_name = f"at-a-station-cross-validation-{c_i}-{i}.p"
            if count == 5:
                with open(checkpoint_file_name, "wb") as fp:
                    pickle.dump(dfs, fp)

                # reset count
                count = 1

            c_i = i

        # df = train(config, train_dataset, test_dataset, round_number=round_number)
        # df = train(*package)

        # Append df to list of dfs
        # dfs.append(df)

        # round_number += 1

    return dfs
    # return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    result = main()
