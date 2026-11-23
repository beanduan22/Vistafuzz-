from .native_base import Runtime


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


RUNTIME = JaxRuntime("jax", "jax.numpy.asarray", "jax.numpy")
