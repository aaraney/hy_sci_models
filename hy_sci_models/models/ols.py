#!/usr/bin/env python3

import pandas as pd
import numpy as np
import statsmodels.api as sm


def OLS(y, X):
    return sm.OLS(y, X)


def train(model):
    return model.fit()


def test(model, X):
    return model.predict(X)