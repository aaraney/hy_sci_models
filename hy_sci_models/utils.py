#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as sm


def summary_stats(y, y_hat):
    mse = sm.mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    r_squared = sm.r2_score(y, y_hat)

    print(f"MSE: {mse}\n" f"RMSE: {rmse}\n" f"R^2: {r_squared}")

    return mse, rmse, r_squared


def percent_bias(y, y_hat):
    return (y_hat - y).sum() / y.sum()


def plot_residuals(
    y: np.array = None, y_hat: np.array = None, residuals: np.array = None
):
    if not residuals:
        residuals = y - y_hat

    f, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(range(len(residuals)), residuals, s=2)
    ax.axhline(y=0, color="black")

    return f, ax


def plot_scatter(y, y_test):
    f, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(y, y_test, s=2)
    ax.axline((1, 1), slope=1, c="black")

    return f, ax


def plot_train_val_loss(training_loss, validation_loss):
    f, ax = plt.subplots(figsize=(10, 10))
    n_epochs = range(1, len(training_loss) + 1)

    ax.plot(n_epochs, training_loss, label="training loss")
    ax.plot(n_epochs, validation_loss, color="red", label="validation loss")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning Rate")
    ax.legend()


def highest_correlation_after_transformation(
    features: pd.DataFrame, label: pd.Series, transformation_mapping: dict
):
    """Determine highest correlation coeficient between features and some label
    from a dictionary of passed in transformation functions.

    Example:
        transformation_map = {
            "inverse": lambda x: 1 /x,
            "log": lambda x: np.log(x),
            "cube_root": lambda x: x ** (1/3)
        }

        highest_correlation_after_transformation(df[['feature_1', 'feature_2', 'feature_3']], df['label'], transformation_map)

    """
    # Empty dataframe
    correlation_df_all_applied_transforms = pd.DataFrame()

    for transformation_key_name, transformation in transformation_mapping.items():
        # Apply each transform to the features then,
        # get the pearsons correlation with the label.
        # Cast the pd.Series to a df and transpose so
        # feature names are column names
        corr_df = pd.DataFrame(features.apply(transformation).corrwith(label)).T

        # Set a col, transformation to the name of the
        # applied transformation, as per the dictionary key
        corr_df["transformation"] = transformation_key_name

        # Concat the resulting dataframe to the empty dataframe
        correlation_df_all_applied_transforms = pd.concat(
            [correlation_df_all_applied_transforms, corr_df]
        )

    # reset the index, droping erroneous prior index
    correlation_df_all_applied_transforms = (
        correlation_df_all_applied_transforms.reset_index(drop=True)
    )

    # make longer. feature col is name of col from input df
    # r is the pearson correlation with the label
    correlation_df_all_long = correlation_df_all_applied_transforms.melt(
        id_vars=["transformation"], value_name="r", var_name="feature"
    )

    # Square to get rid of sign differences and relate correlation magnitudes
    correlation_df_all_long["sq_value"] = correlation_df_all_long["r"] ** 2

    result_df = (
        correlation_df_all_long.groupby("feature")
        # Get rows with the highest R**2, or put differently, have the highest magnitude r value
        .apply(lambda ds: ds[ds["sq_value"] == ds["sq_value"].max()])[
            ["feature", "transformation", "r"]
        ]
    )
    # Sort columns by feature name
    result_df = result_df.iloc[result_df["feature"].str.lower().argsort()].reset_index(
        drop=True
    )
    return result_df