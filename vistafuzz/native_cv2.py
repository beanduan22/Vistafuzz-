from .native_base import Runtime


class OpenCVRuntime(Runtime):
    def array(self, value, dtype_name):
        import numpy as np

        array = np.asarray(value, dtype=dtype_name)
        out = array if array.flags.c_contiguous and array.flags.writeable else np.array(
            array, copy=True, order="C")
        return self.check_array(out, array, dtype_name)


RUNTIME = OpenCVRuntime("cv2", "numpy.asarray")
