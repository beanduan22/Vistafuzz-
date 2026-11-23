from .native_base import Runtime


class OneFlowRuntime(Runtime):
    def array(self, value, dtype_name):
        import numpy as np
        import oneflow as flow

        array = np.asarray(value, dtype=dtype_name)
        native_input = np.array(array, copy=True, order="C")
        dtype = self.dtype(dtype_name) if dtype_name else None
        out = flow.tensor(native_input, dtype=dtype, device=flow.device("cpu"),
                          requires_grad=False)
        return self.check_array(out, array, dtype_name)


RUNTIME = OneFlowRuntime("oneflow", "oneflow.tensor", "oneflow")
