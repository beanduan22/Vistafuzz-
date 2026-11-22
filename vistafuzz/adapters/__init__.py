from __future__ import annotations

import importlib

from .base import Adapter

ALIASES = {"pytorch": "torch", "paddlepaddle": "paddle", "opencv": "cv2",
           "scikit-learn": "sklearn"}
LIBRARIES = ("torch", "tensorflow", "jax", "keras", "paddle", "oneflow",
             "mindspore", "chainer", "numpy", "scipy", "sklearn", "cv2")


def canonical_library(name):
    return ALIASES.get(name.lower(), name)


def get_adapter(api):
    root = canonical_library(api.split(".", 1)[0])
    if root not in LIBRARIES:
        return Adapter(root, "numpy.asarray")
    return importlib.import_module(f"{__name__}.{root}").ADAPTER
