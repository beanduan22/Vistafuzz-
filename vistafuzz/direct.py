from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re

import numpy as np

from . import llm
from .collector import signature_params
from .models import ApiSpec, ParamSpec, Value
from .oracles import _iter_numeric


def prepare(record, config, cache_dir):
    prompt = (
        "Write Python source defining generate_args(rng) for " + record.api + ".\n"
        "The function must return (positional_args, keyword_args), a list and a dict.\n"
        "Use the supplied random.Random instance for randomness and generate valid "
        "inputs across the documented types, shapes, ranges, boundary and non-finite values. "
        "Return source code only, without comments or docstrings. Do not invoke the target API. "
        "Include necessary imports. Do not use a parameter-specification schema.\n"
        "Runtime signature: " + record.api + record.signature_text + "\n"
        "Documentation:\n" + record.doc)
    identity = json.dumps({"prompt": prompt, "model": config.model, "backend": config.backend,
                           "temperature": config.temperature, "num_ctx": config.num_ctx,
                           "max_tokens": config.max_tokens}, sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, record.api + "_" + key + ".py")
    if not os.path.exists(path):
        response = llm.complete(prompt, config)
        fenced = re.search(r"~~~(?:python)?\s*(.*?)~~~", response, re.S)
        if not fenced:
            marker = chr(96) * 3
            fenced = re.search(marker + r"(?:python)?\s*(.*?)" + marker, response, re.S)
        source = fenced.group(1) if fenced else response.strip()
        tree = ast.parse(source)
        if not any(isinstance(n, ast.FunctionDef) and n.name == "generate_args" for n in tree.body):
            raise llm.LLMError("LLM-Direct did not define generate_args(rng)")
        with open(path, "w") as handle:
            handle.write(source + "\n")
    params = [ParamSpec(name=row["name"], required=row["required"], default=row["default"],
                        has_default=row["has_default"], position=row["position"],
                        call_kind=row["call_kind"]) for row in signature_params(record)]
    spec = ApiSpec(api=record.api, doc=record.doc, signature_text=record.signature_text,
                   params=params, extractor=f"llm-direct:{config.backend}:{config.model}",
                   signature_source=record.signature_source)
    return spec, path


def load(path):
    spec = importlib.util.spec_from_file_location("vistafuzz_direct_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "generate_args", None)
    if not callable(fn):
        raise ValueError("missing generate_args(rng)")
    return fn


def describe_value(value):
    if isinstance(value, (list, tuple)):
        recipe = {"form": "sequence", "as_tuple": isinstance(value, tuple),
                  "items": [describe_value(v).recipe for v in value]}
        return Value(kind="shape" if isinstance(value, tuple) else "sequence",
                     payload=value, shape=(len(value),), recipe=recipe, strategy="llm-direct")
    if isinstance(value, dict):
        recipe = {"form": "dict", "items": {k: describe_value(v).recipe for k, v in value.items()}}
        return Value(kind="dict", payload=value, recipe=recipe, strategy="llm-direct")
    if value is None or isinstance(value, (bool, int, float, str, np.generic)):
        value = value.item() if isinstance(value, np.generic) else value
        return Value(kind="default", payload=value, recipe={"form": "const", "value": value},
                     strategy="llm-direct")
    arrays = list(_iter_numeric(value, include_bool=True))
    if len(arrays) != 1:
        raise ValueError(f"cannot serialize LLM-Direct input {type(value).__name__}")
    array = arrays[0]
    return Value(kind="array", payload=value, dtype=str(array.dtype), shape=array.shape,
                 recipe={"form": "literal_array", "dtype": str(array.dtype),
                         "values": array.tolist()}, strategy="llm-direct")


def generate(fn, rng, record, spec):
    generated = fn(rng)
    if not isinstance(generated, tuple) or len(generated) != 2:
        raise ValueError("generate_args(rng) must return (args, kwargs)")
    positional, keyword = generated
    if not isinstance(positional, (list, tuple)) or not isinstance(keyword, dict):
        raise ValueError("generate_args(rng) returned invalid argument containers")
    if record.signature is not None:
        record.signature.bind(*positional, **keyword)
    names = [p.name for p in spec.params if p.call_kind not in ("keyword", "var_keyword")]
    described = {names[i] if i < len(names) else f"arg{i}": describe_value(v)
                 for i, v in enumerate(positional)}
    described.update({name: describe_value(v) for name, v in keyword.items()})
    return list(positional), keyword, described
