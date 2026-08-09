from __future__ import annotations

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from vistafuzz import constraints, generation, materialize, oracles, relationships, synth
from vistafuzz.collector import collect, collect_api, signature_params
from vistafuzz.extraction import extract, extract_offline, parse_llm_params
from vistafuzz.models import (ApiSpec, ParamSpec, Relationship, Value,
                            REL_AXIS_OF, REL_DTYPE_FOLLOWS, REL_SHAPE_FOLLOWS)
from vistafuzz.validation import validate


def test_collect_api_returns_doc_and_signature():
    record = collect_api("numpy.clip")
    assert record is not None
    assert record.doc
    assert "(" in record.signature_text
    names = [row["name"] for row in signature_params(record)]
    assert "a" in names


@pytest.mark.parametrize("api,excluded", [
    ("scipy.test", True),
    ("torch.save", True),
    ("numpy.loadtxt", True),
    ("cv2.imread", True),
    ("torch.cuda.set_device", True),
    ("numpy.random.seed", True),
    ("tensorflow.print", True),
    ("numpy.log", False),
    ("numpy.logaddexp", False),
    ("numpy.trace", False),
    ("scipy.special.logit", False),
    ("scipy.stats.ttest_ind", False),
    ("keras.applications.resnet.preprocess_input", False),
    ("torch.nn.functional.grid_sample", False),
])
def test_name_exclusion_is_token_based(api, excluded):
    from vistafuzz.collector import _is_excluded_name
    assert _is_excluded_name(api) is excluded


def test_collect_discovers_numeric_apis():
    records = collect("numpy", max_depth=1, limit=25)
    assert records
    assert all(r.doc and r.signature_text for r in records)


def test_offline_extraction_aligns_with_signature():
    record = collect_api("numpy.clip")
    specs = extract_offline(record)
    assert [s.name for s in specs][:1] == ["a"]
    assert all(s.position is not None for s in specs)


def test_parse_llm_params_normalizes_relationships():
    payload = {
        "params": [
            {"name": "x", "kind": "tensor", "required": True,
             "dtype_candidates": ["float32"], "shape": {"rank_min": 1, "rank_max": 2}},
            {"name": "y", "kind": "array", "required": True,
             "relationships": [{"kind": "shape_following", "source": "x"},
                               {"kind": "device_following", "source": "x"}]},
            {"name": "axis", "kind": "axis", "required": False, "default": -1,
             "relationships": [{"kind": "axis_of", "source": "x"}]},
        ]
    }
    specs = parse_llm_params(payload)
    assert specs[0].kind == "array"
    assert specs[1].follows_shape_of == "x"
    assert specs[1].relationship("device_following") is None
    assert specs[2].axis_of == "x"


def test_extract_falls_back_when_llm_unreachable():
    from vistafuzz.llm import LLMConfig
    record = collect_api("numpy.clip")
    cfg = LLMConfig(backend="ollama", host="http://127.0.0.1:1", timeout_sec=0.5)
    spec = extract(record, extractor="llm", config=cfg)
    assert spec.extractor == "signature(fallback)"
    assert spec.notes and "llm extraction failed" in spec.notes[0]


def _record_and_spec(api="numpy.clip"):
    record = collect_api(api)
    spec = ApiSpec(api=api, signature_text=record.signature_text, doc=record.doc,
                   params=extract_offline(record))
    return record, spec


def test_validation_runtime_information_wins():
    record, spec = _record_and_spec()
    spec.params[0].required = False
    spec.params[0].call_kind = "keyword"
    spec.params.append(ParamSpec(name="not_a_real_parameter"))
    result = validate(spec, record)
    assert result.ok
    assert "not_a_real_parameter" in result.dropped
    assert result.spec.params[0].required is True
    assert result.spec.param("not_a_real_parameter") is None


def test_validation_drops_relationships_to_missing_sources():
    record, spec = _record_and_spec()
    spec.params[0].relationships = [Relationship(REL_SHAPE_FOLLOWS, "ghost")]
    result = validate(spec, record)
    assert result.ok
    assert result.spec.params[0].relationships == []


def test_envelope_respects_bounds_and_excludes_nonfinite():
    spec = ParamSpec(name="w", kind="number", constraints={"min": 0.0, "max": 1.0})
    env = constraints.envelope_for(spec)
    assert env.low == 0.0 and env.high == 1.0
    assert synth.NONFINITE not in env.patterns


def test_image_like_parameter_gets_image_strategies():
    spec = ParamSpec(name="image", kind="array", doc="Input image.")
    env = constraints.envelope_for(spec)
    assert env.image_like
    assert {synth.NOISE, synth.MASK, synth.DIVISION} <= set(env.patterns)


def test_shape_and_dtype_following_are_resolved():
    x = Value(kind="array", payload=np.zeros((3, 5)), dtype="float64", shape=(3, 5))
    spec = ParamSpec(name="y", kind="array",
                     relationships=[Relationship(REL_SHAPE_FOLLOWS, "x"),
                                    Relationship(REL_DTYPE_FOLLOWS, "x")])
    env = constraints.envelope_for(spec)
    value = generation.make_value(spec, env, {"x": x}, seed=1, dtype="float32",
                                  shape=(9, 9))
    assert value.shape == (3, 5)
    assert value.dtype == "float64"


def test_axis_is_bounded_by_referenced_rank():
    x = Value(kind="array", payload=np.zeros((2, 3, 4)), dtype="float64", shape=(2, 3, 4))
    spec = ParamSpec(name="axis", kind="axis",
                     relationships=[Relationship(REL_AXIS_OF, "x")])
    assert relationships.axis_candidates(spec, {"x": x}) == [-3, -2, -1, 0, 1, 2]
    env = constraints.envelope_for(spec)
    for seed in range(10):
        value = generation.make_value(spec, env, {"x": x}, seed=seed)
        assert -3 <= value.payload <= 2


def test_unsupported_relationship_is_not_enforced():
    spec = ParamSpec(name="y", kind="array",
                     doc="must live on the same device as x",
                     relationships=[])
    env = constraints.envelope_for(spec)
    value = generation.make_value(spec, env, {}, seed=0, shape=(2, 2))
    assert value.shape == (2, 2)


def test_derive_variants_cover_the_three_strategies():
    spec = ParamSpec(name="x", kind="array", dtype_candidates=["float32", "float64"],
                     shape={"rank_min": 1, "rank_max": 3})
    variants = generation.derive_variants(spec, {})
    assert {"type", "size", "value"} <= {v["strategy"] for v in variants}


def test_optional_parameter_can_keep_its_default():
    spec = ParamSpec(name="keepdims", kind="bool", required=False, has_default=True,
                     default=False, enum_values=[True, False])
    variants = generation.derive_variants(spec, {})
    assert any(v.get("omit") for v in variants)


def test_initialize_uses_defaults_for_optional_parameters():
    specs = [ParamSpec(name="x", kind="array", required=True, position=0),
             ParamSpec(name="axis", kind="axis", required=False, has_default=True,
                       default=-1, position=1)]
    args = constraints.initialize(specs)
    assert args["x"].payload is not None
    assert args["axis"].omitted is True


def test_recipe_rebuilds_the_same_value():
    recipe = {"form": "array", "pattern": synth.EXTREME, "dtype": "float64",
              "shape": [2, 3], "seed": 7, "low": None, "high": None, "params": {}}
    first = synth.build_value(recipe)
    second = synth.build_value(json.loads(json.dumps(recipe)))
    assert np.array_equal(first, second, equal_nan=True)


def test_bounds_are_never_violated():
    arr = synth.build_array(synth.EXTREME, "float64", (50,), seed=3, low=0.0, high=1.0)
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def test_materialize_splits_positional_and_keyword():
    specs = [ParamSpec(name="a", kind="array", required=True, position=0,
                       call_kind="positional_or_keyword"),
             ParamSpec(name="b", kind="number", required=False, has_default=True,
                       default=1.0, position=1, call_kind="positional_or_keyword"),
             ParamSpec(name="c", kind="bool", required=False, has_default=True,
                       default=False, position=2, call_kind="keyword")]
    args = {
        "a": Value(kind="array", payload=np.ones((2, 2)), dtype="float64", shape=(2, 2)),
        "b": Value(kind="default", payload=1.0, omitted=True),
        "c": Value(kind="bool", payload=True),
    }
    positional, keyword = materialize.materialize("numpy.add", specs, args)
    assert len(positional) == 1
    assert keyword == {"c": True}


def test_native_array_conversion_falls_back_to_numpy():
    arr = np.arange(6.0).reshape(2, 3)
    assert materialize.to_native_array("numpy", arr, "float64") is arr


@pytest.mark.parametrize("exc_type,message,expected", [
    ("ValueError", "operands could not be broadcast", "rejected"),
    ("TypeError", "unsupported operand", "rejected"),
    ("RuntimeError", "shape mismatch", "rejected"),
    ("RuntimeError", "INTERNAL ASSERT FAILED", "unexpected"),
    ("SystemError", "returned NULL without setting an exception", "unexpected"),
    ("AssertionError", "", "unexpected"),
    ("AssertionError", "expected 4-dimensional tensor, but got 1", "rejected"),
    ("RuntimeError", "CUDA out of memory", "rejected"),
    ("error", "(-5:Bad argument) in function 'add'", "rejected"),
    ("error", "(-215:Assertion failed) _src.depth() == CV_8U", "rejected"),
    ("InvalidArgumentError", "shape must be rank 2", "rejected"),
    ("EnforceNotMet", "Check failed: dims.size() == 2", "unexpected"),
])
def test_exception_classification(exc_type, message, expected):
    assert oracles.classify_exception(exc_type, message) == expected


def test_nan_oracle_flags_non_finite_outputs():
    flagged, detail = oracles.check_nan(np.array([1.0, np.nan]))
    assert flagged and "NaN" in detail
    assert oracles.check_nan(np.array([1.0, 2.0])) == (False, "")


def test_nan_oracle_ignores_integer_and_bool_outputs():
    assert oracles.check_nan(np.array([1, 2, 3])) == (False, "")
    assert oracles.check_nan(True) == (False, "")


def test_inputs_are_finite_detects_injected_nonfinite():
    assert oracles.inputs_are_finite([np.array([1.0, 2.0]), 3.0])
    assert not oracles.inputs_are_finite([np.array([1.0, np.inf])])


def test_crash_verdict():
    assert oracles.crash_verdict(0, None, False) is None
    assert oracles.crash_verdict(-11, 11, False)[0] == "signal"
    assert oracles.crash_verdict(0, None, True)[0] == "hang"


def test_end_to_end_numpy_run(tmp_path):
    from vistafuzz.runner import RunConfig, run
    out = tmp_path / "run"
    summary = run(RunConfig(lib="numpy", out_dir=str(out), budget_sec=2.0,
                            case_timeout=5, extractor="signature",
                            apis=["numpy.clip", "numpy.log", "numpy.divide"]))
    assert summary["apis_tested"] >= 1
    assert summary["cases_generated"] > 0
    assert (out / "summary.json").exists()
    assert (out / "apis.jsonl").exists()


def test_generated_reproducer_runs(tmp_path):
    from vistafuzz.runner import RunConfig, run
    out = tmp_path / "run"
    run(RunConfig(lib="numpy", out_dir=str(out), budget_sec=4.0, case_timeout=5,
                  extractor="signature", apis=["numpy.log", "numpy.divide"]))
    repro_dir = out / "reproducers"
    scripts = sorted(repro_dir.glob("*.py")) if repro_dir.exists() else []
    if not scripts:
        pytest.skip("no finding produced in this short budget")
    proc = subprocess.run([sys.executable, str(scripts[0])], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode in (0, 1), proc.stderr
    assert "calling" in proc.stdout


def test_worker_is_importable_as_module():
    proc = subprocess.run([sys.executable, "-m", "vistafuzz.worker"],
                          capture_output=True, text=True,
                          env={**os.environ,
                               "PYTHONPATH": os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))})
    assert proc.returncode == 2
    assert "usage" in proc.stderr
