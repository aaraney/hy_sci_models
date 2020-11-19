#!/usr/bin/env python3

from setuptools import setup

REQUIRED_PACKAGES = [
    "torch",
    "matplotlib",
    "pandas",
    "numpy",
    "sklearn",
]

setup(
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        "console_scripts": ["hy_sci_models=hy_sci_models.io:main"],
    },
)
