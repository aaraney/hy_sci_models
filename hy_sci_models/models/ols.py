#!/usr/bin/env python3

import pandas as pd
import numpy as np
import statsmodels.api as sm

from dataclasses import dataclass
from .abstractmodeloutput import AbstractModelOutput


def OLS(y, X):
    return sm.OLS(y, X)


def train(model):
    return model.fit()


def test(model, X):
    return model.predict(X)


@dataclass
class OLSModelOutput(AbstractModelOutput):
    X_train: np.array
    X_test: np.array
    y_train: np.array
    y_test: np.array