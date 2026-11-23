from .native_base import Runtime


class ChainerRuntime(Runtime):
    def array(self, value, dtype_name):
        import chainer
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        native_input = np.array(array, copy=True, order="C")
        out = chainer.Variable(native_input, requires_grad=False)
        return self.check_array(out, array, dtype_name)


RUNTIME = ChainerRuntime("chainer", "chainer.Variable", accepts_dtype=False)
