from .native_base import Runtime


class SklearnRuntime(Runtime):
    def array(self, value, dtype_name):
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        out = np.array(array, copy=True, order="C")
        return self.check_array(out, array, dtype_name)


RUNTIME = SklearnRuntime("sklearn", "numpy.asarray")
