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
