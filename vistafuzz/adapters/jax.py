from .base import Adapter


class JaxAdapter(Adapter):
    def array(self, value, dtype_name):
        import jax

        if dtype_name in ("float64", "complex128", "int64", "uint64"):
            jax.config.update("jax_enable_x64", True)
        return super().array(value, dtype_name)


ADAPTER = JaxAdapter("jax", "jax.numpy.asarray", "jax.numpy")
