"""Probe modules for h3_probe. load(name) imports the submodule."""

import importlib


def load(name):
    return importlib.import_module(f"{__name__}.{name}")
