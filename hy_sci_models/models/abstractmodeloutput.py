#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Container


@dataclass
class AbstractModelOutput:
    model: Container