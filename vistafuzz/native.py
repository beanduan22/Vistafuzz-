from __future__ import annotations

import importlib

from .native_base import Runtime

ALIASES = {"pytorch": "torch", "paddlepaddle": "paddle", "opencv": "cv2",
           "scikit-learn": "sklearn"}
LIBRARIES = ("torch", "tensorflow", "jax", "keras", "paddle", "oneflow",
             "mindspore", "chainer", "numpy", "scipy", "sklearn", "cv2")


def canonical_library(name):
    return ALIASES.get(name.lower(), name)


def get_runtime(api):
    root = canonical_library(api.split(".", 1)[0])
    if root not in LIBRARIES:
        return Runtime(root, "numpy.asarray")
    return importlib.import_module(f"{__package__}.native_{root}").RUNTIME
