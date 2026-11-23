from .native_base import Runtime


class PaddleRuntime(Runtime):
    def array(self, value, dtype_name):
        import numpy as np
        import paddle

        array = np.asarray(value, dtype=dtype_name)
        native_input = np.array(array, copy=True, order="C")
        dtype = self.dtype(dtype_name or array.dtype.name)
        if array.dtype.name in ("uint16", "uint32", "uint64"):
            carrier = (native_input.view(np.int64) if array.dtype.name == "uint64"
                       else native_input.astype(np.int64))
            tensor = paddle.to_tensor(carrier, dtype=paddle.int64, place=paddle.CPUPlace(),
                                      stop_gradient=True)
            out = paddle.cast(tensor, dtype)
        else:
            out = paddle.to_tensor(native_input, dtype=dtype, place=paddle.CPUPlace(),
                                   stop_gradient=True)
        return self.check_array(out, array, dtype_name)


RUNTIME = PaddleRuntime("paddle", "paddle.to_tensor", "paddle")
