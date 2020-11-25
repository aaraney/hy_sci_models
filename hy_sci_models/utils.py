#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as sm

# local imports
from . import models


def summary_stats(y, y_hat):
    mse = sm.mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    r_squared = sm.r2_score(y, y_hat)

    print(f"MSE: {mse}\n" f"RMSE: {rmse}\n" f"R^2: {r_squared}")

    return mse, rmse, r_squared


def percent_bias(y, y_hat):
    return (y_hat - y).sum() / y.sum()


def plot_residuals(
    y: np.array = None, y_hat: np.array = None, residuals: np.array = None, ax=None
):
    if ax is not None:
        f = None

    else:
        f, ax = plt.subplots(figsize=(10, 10))

    if not residuals:
        residuals = y - y_hat

    ax.scatter(range(len(residuals)), residuals, s=2)
    ax.axhline(y=0, color="black")

    return f, ax


def plot_scatter(y, y_hat, ax=None):
    if ax is not None:
        f = None

    else:
        f, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(y, y_hat, s=2)
    ax.set_ylabel("y")
    ax.set_xlabel("y_hat")

    ax.axis("square")
    ax.grid()
    ax.axline((1, 1), slope=1, c="black")

    return f, ax


def plot_train_val_loss(training_loss, validation_loss, ax=None):
    if ax is not None:
        f = None

    else:
        f, ax = plt.subplots(figsize=(10, 10))

    n_epochs = range(1, len(training_loss) + 1)

    ax.plot(n_epochs, training_loss, label="training loss")
    ax.plot(n_epochs, validation_loss, color="red", label="validation loss")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning Rate")
    ax.legend()

    return f, ax


def plot_nn_diagnostics(results):
    y, y_hat = models.nn.test(results.model, results.test_loader)

    f, axes = plt.subplots(2, 2, figsize=(10, 10))
    flat_axes = axes.flat

    plot_train_val_loss(results.training_loss, results.validation_loss, ax=flat_axes[0])
    plot_scatter(y, y_hat, ax=flat_axes[1])
    plot_residuals(y=y, y_hat=y_hat, ax=flat_axes[2])

    plt.show()


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


def transform_from_highest_correlation_df(
    features_df: pd.DataFrame, transforms_df: pd.DataFrame, transformation_mapping: dict
):
    """Transform a set of features based on a dataframe of feature names and
    transformations. This method is written be used with the output of and dictionary
    mapping from `highest_correlation_after_transformation`."""

    columns_of_interest = transforms_df["feature"]

    return features_df.apply(
        lambda x: transformation_mapping[
            transforms_df.loc[columns_of_interest == x.name, "transformation"].iat[0]
        ](x)
    )


def _scale_lowest_to_one(ds: pd.Series):
    min = ds.min()

    if min < 0:
        ds = np.abs(min) + ds
    else:
        ds = ds - min

    # Adding 1 to avoid divide by zero
    return ds + 1


def _safety_against_negatives(ds: pd.Series):
    min = ds.min()

    if min <= 0:
        return _scale_lowest_to_one(ds)

    return ds


transformation_functions = {
    "inverse": lambda x: 1 / _safety_against_negatives(x),
    "log": lambda x: np.log(_safety_against_negatives(x)),
    "cube_root": lambda x: x ** (1 / 3),
    "scale_lowest_to_zero": _scale_lowest_to_one,
    "scale_lowest_to_one_log": lambda x: np.log(_scale_lowest_to_one(x)),
    "scale_lowest_to_one_inverse": lambda x: 1 / _scale_lowest_to_one(x),
}
