from .native_base import Runtime


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
        native_input = np.array(array, copy=True, order="C")
        dtype = self.dtype(dtype_name) if dtype_name else None
        out = ms.Tensor(input_data=native_input, dtype=dtype)
        return self.check_array(out, array, dtype_name)


RUNTIME = MindSporeRuntime("mindspore", "mindspore.Tensor", "mindspore")
