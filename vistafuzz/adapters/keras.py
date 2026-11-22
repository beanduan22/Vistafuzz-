from .base import Adapter

ADAPTER = Adapter("keras", "keras.ops.convert_to_tensor", dtype_as_string=True)
