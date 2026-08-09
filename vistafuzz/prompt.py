from __future__ import annotations

import json
from typing import Any

from .models import KINDS, SUPPORTED_RELATIONSHIPS

MAX_DOC_CHARS = 6000

TASK_INSTRUCTION = """\
You convert Python API information into structured parameter specifications for
test-input generation.

Return ONE specification for each parameter of the runtime signature, in the
same order. Follow these requirements:
  * do not generate executable test code, explanations, or prose;
  * preserve information that the documentation actually supports, and leave any
    attribute you cannot justify from the supplied context unspecified (null);
  * align every parameter name with the runtime signature;
  * do not introduce constraints the documentation does not support;
  * normalize an inter-parameter relationship ONLY into one of the supported
    forms below, and otherwise leave it out of `relationships` (it stays in the
    `doc` field);
  * answer with a single JSON object and nothing else.

Supported relationship forms:
  * "shape_follows" - this array/tensor parameter must take the shape of another
    parameter (broadcasting or "same shape as" wording);
  * "axis_of" - this axis-like integer indexes the rank of another parameter;
  * "dtype_follows" - this parameter takes the data type of another parameter.
"""

OUTPUT_SCHEMA = {
    "params": [
        {
            "name": "<parameter name from the runtime signature>",
            "kind": f"one of {list(KINDS)}",
            "required": "true|false",
            "default": "<default value, or null>",
            "dtype_candidates": ["float32", "float64", "int32", "..."],
            "shape": {
                "rank_min": "int|null",
                "rank_max": "int|null",
                "dims": "[int,...]|null   # only when the doc fixes concrete dimensions",
            },
            "constraints": {
                "min": "number|null",
                "max": "number|null",
                "exclusive_min": "true|false|null",
                "exclusive_max": "true|false|null",
                "allow_nonfinite": "true|false|null",
                "len_min": "int|null",
                "len_max": "int|null",
            },
            "enum_values": ["<allowed literal>", "..."],
            "relationships": [
                {"kind": f"one of {list(SUPPORTED_RELATIONSHIPS)}",
                 "source": "<other parameter name>"}
            ],
            "doc": "<one-line summary, including any relationship that could NOT be normalized>",
        }
    ]
}

EXAMPLES: list[dict[str, Any]] = [
    {
        "api": "lib.reduce_along(x, axis=None, dtype=None, keepdims=False)",
        "doc": ("x : array_like\n    Input array.\n"
                "axis : int, optional\n    Axis along which to operate. Must be in "
                "range [-x.ndim, x.ndim).\n"
                "dtype : data-type, optional\n    Type of the returned array.\n"
                "keepdims : bool, optional\n    If True, the reduced axis is kept."),
        "output": {
            "params": [
                {"name": "x", "kind": "array", "required": True, "default": None,
                 "dtype_candidates": ["float32", "float64"],
                 "shape": {"rank_min": 1, "rank_max": 4, "dims": None},
                 "constraints": {}, "enum_values": [], "relationships": [],
                 "doc": "Input array."},
                {"name": "axis", "kind": "axis", "required": False, "default": None,
                 "dtype_candidates": ["int64"], "shape": {},
                 "constraints": {}, "enum_values": [],
                 "relationships": [{"kind": "axis_of", "source": "x"}],
                 "doc": "Axis in range [-x.ndim, x.ndim)."},
                {"name": "dtype", "kind": "dtype", "required": False, "default": None,
                 "dtype_candidates": ["float32", "float64"], "shape": {},
                 "constraints": {}, "enum_values": [], "relationships": [],
                 "doc": "Type of the returned array."},
                {"name": "keepdims", "kind": "bool", "required": False, "default": False,
                 "dtype_candidates": [], "shape": {}, "constraints": {},
                 "enum_values": [True, False], "relationships": [],
                 "doc": "Keep the reduced axis."},
            ]
        },
    },
    {
        "api": "lib.blend(a, b, weight=0.5)",
        "doc": ("a : tensor\n    First input.\n"
                "b : tensor\n    Second input, same shape and dtype as `a`.\n"
                "weight : float\n    Blending weight in [0, 1]. Must be finite.\n"
                "The operation is only defined for inputs on the same device."),
        "output": {
            "params": [
                {"name": "a", "kind": "array", "required": True, "default": None,
                 "dtype_candidates": ["float32", "float64"],
                 "shape": {"rank_min": 1, "rank_max": 4, "dims": None},
                 "constraints": {}, "enum_values": [], "relationships": [],
                 "doc": "First input."},
                {"name": "b", "kind": "array", "required": True, "default": None,
                 "dtype_candidates": ["float32", "float64"], "shape": {},
                 "constraints": {}, "enum_values": [],
                 "relationships": [{"kind": "shape_follows", "source": "a"},
                                   {"kind": "dtype_follows", "source": "a"}],
                 "doc": "Second input, same shape and dtype as a."},
                {"name": "weight", "kind": "number", "required": False, "default": 0.5,
                 "dtype_candidates": ["float64"], "shape": {},
                 "constraints": {"min": 0.0, "max": 1.0, "allow_nonfinite": False},
                 "enum_values": [], "relationships": [],
                 "doc": ("Blending weight in [0, 1]; the same-device requirement is "
                         "not a supported relationship form.")},
            ]
        },
    },
]


def _fmt_signature_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(runtime signature could not be introspected)"
    lines = []
    for row in rows:
        default = "<required>" if row["required"] else repr(row.get("default"))
        lines.append(f"  - {row['name']}: position={row['position']}, "
                     f"call_kind={row['call_kind']}, required={row['required']}, "
                     f"default={default}, annotation={row.get('annotation') or '-'}")
    return "\n".join(lines)


def build_prompt(api: str, signature_text: str, doc: str,
                 signature_rows: list[dict[str, Any]]) -> str:
    doc = (doc or "").strip()
    if len(doc) > MAX_DOC_CHARS:
        doc = doc[:MAX_DOC_CHARS] + "\n... [documentation truncated]"

    examples = []
    for ex in EXAMPLES:
        examples.append(
            f"API: {ex['api']}\nDocumentation:\n{ex['doc']}\n"
            f"Output:\n{json.dumps(ex['output'], indent=2)}"
        )

    return f"""### Task
{TASK_INSTRUCTION}

### API context
API name: {api}
Runtime signature: {api}{signature_text or '(...)'}
Signature-aligned parameters:
{_fmt_signature_rows(signature_rows)}

Raw documentation:
\"\"\"
{doc or '(no documentation)'}
\"\"\"

### Output schema
{json.dumps(OUTPUT_SCHEMA, indent=2)}

### Examples
{chr(10).join(examples)}

### Now produce the JSON object for {api}
"""
