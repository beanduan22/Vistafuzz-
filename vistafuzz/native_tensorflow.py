from .native_base import Runtime


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


RUNTIME = TensorFlowRuntime("tensorflow", "tensorflow.convert_to_tensor", "tensorflow")
