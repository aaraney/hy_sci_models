#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from typing import List, Tuple, Union

# local imports
from .models.nn import SingleLayerNet

# SEED variable. This is changed if passed in setup file
SEED = None


def setup(dataset: str, features: List[str], labels: str) -> Tuple[pd.DataFrame]:
    """Read input pickle file, extract features and labels of interest, pre-process
    (downcast to float32) and return a tuple of features, labels.
    """
    # Read input file
    data = pd.read_pickle(dataset)

    # NOTE: DELETE SOON
    data = data[(data[labels] > 0.0) & (data["site_no"] != 13290450)]

    # Downcast all float64 to float32
    data = downcast_columns_to_float32(data)

    # Extract labels
    # NOTE: This needs to change. This should be changed in `models.nn::test` also
    processed_labels = np.log(data[labels] + 1)

    # Process input features by taking subset and filling na values with the mean from
    # the column of origin
    processed_feature_data = process_features(data, features)

    return processed_feature_data, processed_labels


class NNDataset(Dataset):
    """ Simple Dataset class """

    def __init__(self, X, y) -> None:
        self.X_data = X if torch.is_tensor(X) else torch.from_numpy(X)
        self.y_data = y if torch.is_tensor(y) else torch.from_numpy(y)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def generate_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test, training_batchsize
) -> Tuple[NNDataset]:
    """
    Returns training, testing, and validation `NNDataset`'s

    returns train_loader, test_loader, val_loader
    """
    # Initialize dataset objects
    train_dataset = NNDataset(X_train, y_train)
    val_dataset = NNDataset(X_val, y_val)
    test_dataset = NNDataset(X_test, y_test)

    # Initialize dataloader
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=training_batchsize, shuffle=True
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    return train_loader, val_loader, test_loader


def split_dataset_into_train_test_val(
    X: pd.DataFrame, y: pd.DataFrame, train_size: float
) -> Tuple[np.array]:
    """Returned as X_train, X_val, X_test, y_train, y_val, y_test

    test and validation are always (1 - train_size) / 2
    """

    if train_size >= 1:
        error_message = "training size cannot greater than or equal to 1"
        raise ValueError(error_message)

    first_split_size = 1 - train_size

    # test_size & val_size = (1 - train) * 0.5
    TEST_SIZE = 0.5

    # global var, SEED, is default None
    # split features and label data into training and testing set
    X_train, X_first_spite, y_train, y_first_split = train_test_split(
        X, y, test_size=first_split_size, random_state=SEED
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_first_spite, y_first_split, test_size=TEST_SIZE, random_state=SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_and_transform(X_train, X_val, X_test, y_train, y_val, y_test):
    """Returned as X_train, X_val, X_test, y_train, y_val, y_test"""
    # Scaling input training set
    scaler = MinMaxScaler()

    # Fit and transform training set
    X_train = scaler.fit_transform(X_train)

    # Transform testing dataset using the fit of the training set
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Potentially unnecessary casting?
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_test), np.array(y_test)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def downcast_columns_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.columns[df.dtypes.eq(np.number)]].apply(
        pd.to_numeric, downcast="float"
    )


def process_features(
    df: pd.DataFrame, features_of_interest: Union[List[str], None] = None
) -> pd.DataFrame:
    """Extract features of intrest from input dataframe and fill na values with the
    mean from its given column"""
    if features_of_interest:
        df = df[features_of_interest]

    return df.fillna(df.mean())


if __name__ == "__main__":
    # This is the _functional_ version. Leaving for future sake
    EPOCHS = 250
    BATCH_SIZE = 150
    LEARNING_RATE = 0.0001

    INPUT_DATASET_PATH = "../data/joined_data_subset.p"

    FEATURES_OF_INTEREST = [
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
        "HW",
        "DOR",
        "width_m",  # peirong width
        # "stream_wdth_va", # swot measured width
        # "drain_area_va",
        # "contrib_drain_area_va",
        # "alt_va",
    ]

    LABELS = "mean_depth_va"
    # [
    #     "mean_depth_va",
    #     # "max_depth_va"
    # ]

    # Read input file
    data = pd.read_pickle(INPUT_DATASET_PATH)

    # DELETE SOON
    data = data[(data[LABELS] > 0.0) & (data["site_no"] != 13290450)]

    # Downcast all float64 to float32
    data = downcast_columns_to_float32(data)

    # Extract labels
    labels = np.log(data[LABELS] + 1)

    # Process input features by taking subset and filling na values with the mean from
    # the column of origin
    processed_feature_data = process_features(data, FEATURES_OF_INTEREST)

    # Number of neurons in input layer
    N_INPUT_FEATURES = processed_feature_data.shape[1]

    # split features and label data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        processed_feature_data, labels, test_size=0.2, random_state=419
    )

    # Scaling input training set
    scaler = MinMaxScaler()

    # Fit and transform training set
    X_train = scaler.fit_transform(X_train)

    # Transform testing dataset using the fit of the training set
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Initialize dataset objects
    train_dataset = NNDataset(X_train, y_train)
    test_dataset = NNDataset(X_test, y_test)

    # Initialize dataloader
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    # Retrieve device information
    device = torch.device("cpu")

    # Define NN model
    model = SingleLayerNet(N_INPUT_FEATURES)
    model.to(device)

    # Initialize criterion function
    criterion = nn.MSELoss()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ########### Training ###########
    training_stats_list = []

    print("Beginning Training")

    for e in range(EPOCHS):

        train_epoch_loss = 0

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

        print(f"Epoch loss: {train_epoch_loss / len(train_loader)}")

    # Save the model to use it later
    # torch.save(model, "../../models/ann_1_32_1.pt")

    y_pred_list = []

    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list.append(y_test_pred.cpu().numpy())

    y_pred_list = np.concatenate(y_pred_list).squeeze()

    y_pred_list = (np.e ** y_pred_list) - 1
    y_test = (np.e ** y_test) - 1

    mse = mean_squared_error(y_test, y_pred_list)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred_list)

    print(f"\nMean Squared Error : {mse}")
    print(f"RMSE: {rmse}")
    print(f"R^2 : {r_squared}")

    # # Create a plot of y vs yhat and a residuals plots
    f, axes = plt.subplots(2, 1, figsize=(10, 20))

    axes.flat[0].scatter(y_test, y_pred_list, s=2)
    axes.flat[0].axline((1, 1), slope=1, c="black")

    residuals = y_test - y_pred_list
    axes.flat[1].scatter(range(len(residuals)), residuals, s=2)

    plt.show()