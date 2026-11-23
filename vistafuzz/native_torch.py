from .native_base import Runtime


class TorchRuntime(Runtime):
    def signature(self, obj):
        signature = super().signature(obj)
        if signature is not None and (not signature.parameters or any(
                p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values())):
            return signature
        import inspect

        schema = getattr(obj, "_schema", None)
        if schema is None or getattr(schema, "is_vararg", False):
            return None
        params = []
        for argument in schema.arguments:
            kind = (inspect.Parameter.KEYWORD_ONLY if argument.kwarg_only
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD)
            default = (argument.default_value if argument.has_default_value()
                       else inspect.Parameter.empty)
            params.append(inspect.Parameter(argument.name, kind, default=default))
        try:
            return inspect.Signature(params)
        except ValueError:
            return None

    def array(self, value, dtype_name):
        import numpy as np
        import torch

        array = np.asarray(value, dtype=dtype_name)
        native_input = np.array(array, copy=True, order="C")
        dtype = self.dtype(dtype_name) if dtype_name else None
        out = torch.tensor(native_input, dtype=dtype, device="cpu", requires_grad=False)
        return self.check_array(out, array, dtype_name)


RUNTIME = TorchRuntime("torch", "torch.tensor", "torch")
