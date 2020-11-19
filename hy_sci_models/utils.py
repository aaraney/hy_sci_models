#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score


def summary_stats(y, y_hat):
    mse = mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y, y_hat)

    print(f"MSE: {mse}\n" f"RMSE: {rmse}\n" f"R^2: {r_squared}")

    return mse, rmse, r_squared


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

    ax.scatter(y_test, y_pred_list, s=2)
    ax.axline((1, 1), slope=1, c="black")

    return f, ax
