#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Callable, Dict, Type


class AbstractModelConfig(dataclass):
    # workflow function for carrying model fitting
    workflow: Callable

    # keys that must be present in config for the method to work and their expected
    # type.
    # e.g. {"batch_size": int}
    major_keys: Dict[str, Type]

    # optional keys present in config and their expected type
    minor_keys: Dict[str, Type]
