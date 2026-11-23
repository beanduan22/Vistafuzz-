from .native_base import Runtime


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
                native_dtype = getattr(torch, dtype, None)
                if native_dtype is None:
                    raise TypeError(f"the installed PyTorch does not support {dtype}")
                native_input = np.array(array, copy=True, order="C")
                out = torch.tensor(native_input, dtype=native_dtype, device="cpu",
                                   requires_grad=False)
                return self.check_array(out, array, dtype_name)
        with context:
            out = keras.ops.convert_to_tensor(array, dtype=dtype, sparse=False, ragged=False)
        return self.check_array(out, array, dtype_name)


RUNTIME = KerasRuntime("keras", "keras.ops.convert_to_tensor", dtype_as_string=True)
