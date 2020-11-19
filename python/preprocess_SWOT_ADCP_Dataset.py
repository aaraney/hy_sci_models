#!/usr/bin/env python3
import pandas as pd
import numpy as np
import geopandas as gpd

import argparse
from pathlib import Path


def is_file(_file):
    if Path(_file).is_file() and Path(_file).exists():
        return str(_file.resolve())
    raise FileNotFoundError("The provided file does not exist")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=is_file,
        nargs=1,
        help="Input shapefile",
        required=True,
    )
    return parser


# gpd.
# FILE = Path()
# gpd.datasets

if __name__ == "__main__":
    args = cli()

    args.parse_args()
    input_file = args.input_file
    pass