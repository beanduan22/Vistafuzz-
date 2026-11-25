"""Per-library runtimes: how VistaFuzz turns a generated NumPy value into each
library's tensor type, reads its docs/signatures, and resolves dtypes."""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

import numpy as np


def resolve_object(path):
    parts = path.split(".")
    for size in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:size]))
            break
        except ModuleNotFoundError as exc:
            if exc.name and not ".".join(parts[:size]).startswith(exc.name):
                raise
    else:
        raise ImportError(path)
    for name in parts[size:]:
        obj = getattr(obj, name)
    return obj


@dataclass(frozen=True)
class Runtime:
    root: str
    array_path: str
    dtype_path: str = "numpy"
    dtype_as_string: bool = False
    accepts_dtype: bool = True

    def documentation(self, obj):
        return inspect.getdoc(obj) or ""

    def signature(self, obj):
        try:
            return inspect.signature(obj)
        except (TypeError, ValueError):
            return None

    def dtype(self, name):
        if self.dtype_as_string:
            return name
        namespace = importlib.import_module(self.dtype_path)
        if self.dtype_path == "numpy":
            return np.dtype(name)
        candidate = getattr(namespace, name, None)
        if candidate is None:
            raise TypeError(f"{self.root} does not support dtype {name}")
        return candidate

    def array(self, value, dtype_name):
        ctor = resolve_object(self.array_path)
        dtype = self.dtype(dtype_name) if dtype_name else None
        array = np.asarray(value, dtype=dtype_name)
        out = ctor(array, dtype=dtype) if self.accepts_dtype else ctor(array)
        return self.check_array(out, array, dtype_name)

    def check_array(self, out, array, dtype_name):
        shape = getattr(out, "shape", None)
        if shape is None:
            raise TypeError(f"{self.root} did not expose the materialized shape")
        if tuple(shape) != array.shape:
            raise ValueError(f"{self.root} changed the generated shape")
        actual = getattr(out, "dtype", None)
        if actual is None:
            raise TypeError(f"{self.root} did not expose the materialized dtype")
        expected = np.dtype(dtype_name if dtype_name is not None else array.dtype)
        named_dtype = getattr(actual, "name", None)
        if isinstance(named_dtype, str):
            named_dtype = named_dtype.lower()
        candidates = (getattr(actual, "as_numpy_dtype", None), named_dtype,
                      actual, str(actual).rsplit(".", 1)[-1].lower())
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                observed = np.dtype(candidate)
            except (TypeError, ValueError):
                continue
            if observed != expected:
                raise ValueError(f"{self.root} changed {expected.name} to {actual}")
            return out
        raise TypeError(f"{self.root} exposed an unrecognized dtype: {actual}")


# --- torch -----------------------------------------------------------------
class TorchRuntime(Runtime):
    def signature(self, obj):
        signature = super().signature(obj)
        if signature is not None and (not signature.parameters or any(
                p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values())):
            return signature
        import inspect

        schema = getattr(obj, "_schema", None)
        if schema is None or getattr(schema, "is_vararg", False):
            return None
        params = []
        for argument in schema.arguments:
            kind = (inspect.Parameter.KEYWORD_ONLY if argument.kwarg_only
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD)
            default = (argument.default_value if argument.has_default_value()
                       else inspect.Parameter.empty)
            params.append(inspect.Parameter(argument.name, kind, default=default))
        try:
            return inspect.Signature(params)
        except ValueError:
            return None

    def array(self, value, dtype_name):
        import numpy as np
        import torch

        array = np.asarray(value, dtype=dtype_name)
        host_input = np.array(array, copy=True, order="C")
        dtype = self.dtype(dtype_name) if dtype_name else None
        out = torch.tensor(host_input, dtype=dtype, device="cpu", requires_grad=False)
        return self.check_array(out, array, dtype_name)


_TORCH = TorchRuntime("torch", "torch.tensor", "torch")


# --- tensorflow ------------------------------------------------------------
class TensorFlowRuntime(Runtime):
    def documentation(self, obj):
        doc = super().documentation(obj)
        return doc or super().documentation(getattr(obj, "python_function", None))

    def signature(self, obj):
        import inspect

        function = getattr(obj, "python_function", obj)
        signature = super().signature(function)
        if signature is None or all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                                    for p in signature.parameters.values()):
            try:
                from tensorflow.python.util import tf_decorator

                _, target = tf_decorator.unwrap(obj)
                return inspect.signature(target)
            except (TypeError, ValueError):
                return signature
        return signature

    def dtype(self, name):
        import tensorflow as tf

        return tf.dtypes.as_dtype(name)

    def array(self, value, dtype_name):
        import numpy as np
        import tensorflow as tf

        array = np.asarray(value, dtype=dtype_name)
        dtype = self.dtype(dtype_name) if dtype_name else None
        with tf.device("/CPU:0"):
            out = tf.convert_to_tensor(array, dtype=dtype)
        return self.check_array(out, array, dtype_name)


_TENSORFLOW = TensorFlowRuntime("tensorflow", "tensorflow.convert_to_tensor", "tensorflow")


# --- jax -------------------------------------------------------------------
class JaxRuntime(Runtime):
    def documentation(self, obj):
        doc = super().documentation(obj)
        wrapped = getattr(obj, "__wrapped__", None)
        return doc or super().documentation(wrapped)

    def dtype(self, name):
        import jax
        import numpy as np

        dtype = np.dtype(name)
        if dtype.name in ("float64", "complex128", "int64", "uint64"):
            jax.config.update("jax_enable_x64", True)
        return dtype

    def array(self, value, dtype_name):
        import jax
        import jax.numpy as jnp
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        dtype = self.dtype(dtype_name or str(array.dtype))
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            out = jnp.asarray(array, dtype=dtype)
        out.block_until_ready()
        return self.check_array(out, array, dtype_name)


_JAX = JaxRuntime("jax", "jax.numpy.asarray", "jax.numpy")


# --- keras -----------------------------------------------------------------
class KerasRuntime(Runtime):
    def dtype(self, name):
        import keras

        return keras.backend.standardize_dtype(name)

    def array(self, value, dtype_name):
        from contextlib import nullcontext
        import keras
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        dtype = self.dtype(dtype_name) if dtype_name else str(array.dtype)
        backend = keras.backend.backend()
        context = nullcontext()
        if backend == "jax":
            import jax

            if dtype in ("float64", "complex128", "int64", "uint64"):
                jax.config.update("jax_enable_x64", True)
            context = jax.default_device(jax.devices("cpu")[0])
        elif backend == "tensorflow":
            import tensorflow as tf

            context = tf.device("/CPU:0")
        elif backend == "torch":
            import torch

            context = torch.device("cpu")
            if dtype in ("uint16", "uint32", "uint64"):
                lib_dtype = getattr(torch, dtype, None)
                if lib_dtype is None:
                    raise TypeError(f"the installed PyTorch does not support {dtype}")
                host_input = np.array(array, copy=True, order="C")
                out = torch.tensor(host_input, dtype=lib_dtype, device="cpu",
                                   requires_grad=False)
                return self.check_array(out, array, dtype_name)
        with context:
            out = keras.ops.convert_to_tensor(array, dtype=dtype, sparse=False, ragged=False)
        return self.check_array(out, array, dtype_name)


_KERAS = KerasRuntime("keras", "keras.ops.convert_to_tensor", dtype_as_string=True)


# --- paddle ----------------------------------------------------------------
class PaddleRuntime(Runtime):
    def array(self, value, dtype_name):
        import numpy as np
        import paddle

        array = np.asarray(value, dtype=dtype_name)
        host_input = np.array(array, copy=True, order="C")
        dtype = self.dtype(dtype_name or array.dtype.name)
        if array.dtype.name in ("uint16", "uint32", "uint64"):
            carrier = (host_input.view(np.int64) if array.dtype.name == "uint64"
                       else host_input.astype(np.int64))
            tensor = paddle.to_tensor(carrier, dtype=paddle.int64, place=paddle.CPUPlace(),
                                      stop_gradient=True)
            out = paddle.cast(tensor, dtype)
        else:
            out = paddle.to_tensor(host_input, dtype=dtype, place=paddle.CPUPlace(),
                                   stop_gradient=True)
        return self.check_array(out, array, dtype_name)


_PADDLE = PaddleRuntime("paddle", "paddle.to_tensor", "paddle")


# --- oneflow ---------------------------------------------------------------
class OneFlowRuntime(Runtime):
    def array(self, value, dtype_name):
        import numpy as np
        import oneflow as flow

        array = np.asarray(value, dtype=dtype_name)
        host_input = np.array(array, copy=True, order="C")
        dtype = self.dtype(dtype_name) if dtype_name else None
        out = flow.tensor(host_input, dtype=dtype, device=flow.device("cpu"),
                          requires_grad=False)
        return self.check_array(out, array, dtype_name)


_ONEFLOW = OneFlowRuntime("oneflow", "oneflow.tensor", "oneflow")


# --- mindspore -------------------------------------------------------------
class MindSporeRuntime(Runtime):
    def dtype(self, name):
        from mindspore import dtype as mstype

        candidate = getattr(mstype, name, None)
        if candidate is None:
            raise TypeError(f"mindspore does not support dtype {name}")
        return candidate

    def array(self, value, dtype_name):
        import mindspore as ms
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        host_input = np.array(array, copy=True, order="C")
        dtype = self.dtype(dtype_name) if dtype_name else None
        out = ms.Tensor(input_data=host_input, dtype=dtype)
        return self.check_array(out, array, dtype_name)


_MINDSPORE = MindSporeRuntime("mindspore", "mindspore.Tensor", "mindspore")


# --- chainer ---------------------------------------------------------------
class ChainerRuntime(Runtime):
    def array(self, value, dtype_name):
        import chainer
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        host_input = np.array(array, copy=True, order="C")
        out = chainer.Variable(host_input, requires_grad=False)
        return self.check_array(out, array, dtype_name)


_CHAINER = ChainerRuntime("chainer", "chainer.Variable", accepts_dtype=False)


# --- numpy -----------------------------------------------------------------
_NUMPY = Runtime("numpy", "numpy.asarray")


# --- scipy -----------------------------------------------------------------
_SCIPY = Runtime("scipy", "numpy.asarray")


# --- sklearn ---------------------------------------------------------------
class SklearnRuntime(Runtime):
    def array(self, value, dtype_name):
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        out = np.array(array, copy=True, order="C")
        return self.check_array(out, array, dtype_name)


_SKLEARN = SklearnRuntime("sklearn", "numpy.asarray")


# --- cv2 -------------------------------------------------------------------
class OpenCVRuntime(Runtime):
    def array(self, value, dtype_name):
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        out = array if array.flags.c_contiguous and array.flags.writeable else np.array(
            array, copy=True, order="C")
        return self.check_array(out, array, dtype_name)


_CV2 = OpenCVRuntime("cv2", "numpy.asarray")


RUNTIMES = {
    "torch": _TORCH,
    "tensorflow": _TENSORFLOW,
    "jax": _JAX,
    "keras": _KERAS,
    "paddle": _PADDLE,
    "oneflow": _ONEFLOW,
    "mindspore": _MINDSPORE,
    "chainer": _CHAINER,
    "numpy": _NUMPY,
    "scipy": _SCIPY,
    "sklearn": _SKLEARN,
    "cv2": _CV2,
}

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
    return RUNTIMES[root]
