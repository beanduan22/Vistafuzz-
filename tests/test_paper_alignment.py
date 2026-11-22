from __future__ import annotations

import ast
import csv
import io
import json
import random
import subprocess
import sys
import tokenize

import numpy as np
import pytest

from vistafuzz import constraints, generation, oracles, relationships, reproducer, synth
from vistafuzz.adapters import LIBRARIES, get_adapter
from vistafuzz.collector import ApiRecord, collect_api
from vistafuzz.extraction import extract
from vistafuzz.llm import LLMConfig
from vistafuzz.models import ApiSpec, Finding, ParamSpec, Relationship, Value
from vistafuzz.prompt import build_prompt
from vistafuzz.runner import RunConfig, run, run_api
from vistafuzz.validation import validate


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "uint8", "uint64"])
@pytest.mark.parametrize("exclusive", [False, True])
def test_integer_domain_is_preserved_after_dtype_conversion(dtype, exclusive):
    spec = ParamSpec(name="n", kind="int", dtype_candidates=[dtype],
                     constraints={"min": 0, "max": 3, "exclusive_min": exclusive})
    env = constraints.envelope_for(spec)
    for pattern in env.patterns:
        for seed in range(20):
            value = generation.make_value(spec, env, {}, seed=seed, pattern=pattern)
            assert value is not None
            assert (0 < value.payload if exclusive else 0 <= value.payload)
            assert value.payload <= 3


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("pattern", [synth.ORDINARY, synth.EXTREME, synth.BOUNDARY, synth.NONFINITE])
def test_finite_only_values_remain_finite_in_the_selected_dtype(dtype, pattern):
    spec = ParamSpec(name="x", kind="array", dtype_candidates=[dtype],
                     constraints={"allow_nonfinite": False}, shape={"dims": [4, 8]})
    value = generation.make_value(spec, constraints.envelope_for(spec), {},
                                  seed=0, pattern=pattern)
    assert value is not None
    assert value.payload.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(value.payload))


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_open_float_interval_is_preserved_in_actual_dtype(dtype):
    value = synth.build_array(synth.BOUNDARY, dtype, (200,), 5, low=0, high=1,
                               exclusive_min=True, exclusive_max=True)
    assert np.all(value > 0)
    assert np.all(value < 1)


def test_impossible_integer_domain_is_not_replaced_with_an_invalid_value():
    spec = ParamSpec(name="n", kind="int", constraints={"min": 0, "max": 1,
                                                       "exclusive_min": True, "exclusive_max": True})
    assert generation.make_value(spec, constraints.envelope_for(spec), {}, seed=0) is None


@pytest.mark.parametrize("shape", [(), (0,), (500, 500), (1, 1, 1, 1)])
def test_documented_fixed_shapes_are_preserved(shape):
    spec = ParamSpec(name="x", kind="array", shape={"dims": list(shape)})
    env = constraints.envelope_for(spec)
    assert env.shapes == [shape]
    value = generation.make_value(spec, env, {}, seed=0)
    assert value.payload.shape == shape


def test_fixed_rank_four_is_generated_as_rank_four():
    spec = ParamSpec(name="x", kind="array", shape={"rank_min": 4, "rank_max": 4})
    value = generation.make_value(spec, constraints.envelope_for(spec), {}, seed=0)
    assert value.payload.ndim == 4


def test_unsupported_fixed_shape_is_skipped_instead_of_changed():
    spec = ParamSpec(name="x", kind="array", shape={"dims": [1000000, 1000000]})
    assert generation.make_value(spec, constraints.envelope_for(spec), {}, seed=0) is None


@pytest.mark.parametrize("kind", ["sequence", "shape"])
@pytest.mark.parametrize("length", [0, 1, 5])
def test_structured_argument_length_is_preserved(kind, length):
    spec = ParamSpec(name="s", kind=kind, constraints={"len_min": length, "len_max": length})
    env = constraints.envelope_for(spec)
    for seed in range(10):
        value = generation.make_value(spec, env, {}, seed=seed)
        assert len(value.payload) == length
        assert generation.constraints_satisfied([spec], {"s": value})


def test_dictionary_argument_is_a_dictionary():
    spec = ParamSpec(name="options", kind="dict")
    value = generation.make_value(spec, constraints.envelope_for(spec), {}, seed=0)
    assert isinstance(value.payload, dict)
    assert synth.build_value(value.recipe) == {}


def test_reverse_order_dependency_chain_is_resolved():
    specs = [
        ParamSpec(name="a", kind="array", relationships=[Relationship("shape_follows", "b")]),
        ParamSpec(name="b", kind="array", relationships=[Relationship("shape_follows", "c")]),
        ParamSpec(name="c", kind="array"),
    ]
    args = {name: Value(kind="array", dtype="float64", shape=shape,
                        recipe={"pattern": "ordinary"})
            for name, shape in [("a", (1,)), ("b", (4,)), ("c", (2, 3))]}
    generation.refresh_dependents(specs, args, random.Random(0))
    assert args["a"].shape == args["b"].shape == args["c"].shape == (2, 3)


def test_cyclic_relationships_are_rejected_before_generation():
    record = collect_api("numpy.add")
    spec = extract(record, extractor="signature")
    spec.params[0].relationships = [Relationship("shape_follows", spec.params[1].name)]
    spec.params[1].relationships = [Relationship("shape_follows", spec.params[0].name)]
    result = validate(spec, record)
    assert not result.ok
    assert "cyclic" in result.reason


def test_absent_signature_cannot_validate_a_hallucinated_parameter():
    record = ApiRecord(api="fake.api", obj=None, doc="Numerical function", signature_text="",
                       signature=None, module="fake")
    result = validate(ApiSpec(api="fake.api", params=[ParamSpec(name="invented")]), record)
    assert not result.ok


def test_gelu_paper_output_is_checked_with_nonfinite_inputs():
    x = np.array([[np.nan, np.inf, -np.inf], [0.1, 0.2, 0.3]])
    y = np.array([[np.nan, np.nan, np.nan], [0.1, 0.2, 0.3]])
    assert not oracles.inputs_are_finite([x])
    assert oracles.check_nan(y, [x])[0]


def test_matching_nonfinite_propagation_is_not_a_new_anomaly():
    x = np.array([np.nan, np.inf, -np.inf])
    assert oracles.check_nan(x.copy(), [x]) == (False, "")


@pytest.mark.parametrize("output", [True, 1.0, np.array([False]), np.array([0.0])])
def test_documented_nan_propagation_detects_finite_and_boolean_outputs(output):
    assert oracles.check_nan(output, [np.array([np.nan])], "NaNs are propagated.")[0]


def test_nan_ignoring_behavior_does_not_require_propagation():
    assert oracles.check_nan(True, [np.nan], "NaNs are ignored and do not propagate.") == (False, "")


def test_corner_subpix_assertion_is_reported_for_constraint_satisfying_inputs():
    message = ("(-215:Assertion failed) Rect(0, 0, src.cols, src.rows).contains(cT) "
               "in function 'cv::cornerSubPix'")
    assert oracles.classify_exception("error", message, constraints_satisfied=True) == "unexpected"
    assert oracles.classify_exception("error", message, constraints_satisfied=False) == "rejected"


def test_nested_outputs_are_not_silently_truncated():
    output = {"values": [np.array([1.0]) for _ in range(12)] + [np.array([np.nan])]}
    assert oracles.check_nan(output)[0]


def test_specification_cache_reuses_extraction_and_invalidates_model_changes(monkeypatch, tmp_path):
    calls = []
    def complete(prompt, config):
        calls.append(config.model)
        return json.dumps({"params": [{"name": "x", "kind": "array", "required": True}]})
    monkeypatch.setattr("vistafuzz.llm.complete", complete)
    record = collect_api("numpy.negative")
    first = extract(record, config=LLMConfig(), cache_dir=str(tmp_path))
    first.params[0].kind = "str"
    second = extract(record, config=LLMConfig(), cache_dir=str(tmp_path))
    assert second.params[0].kind == "array"
    extract(record, config=LLMConfig(model="qwen2.5-coder:7b"), cache_dir=str(tmp_path))
    assert calls == ["qwen2.5-coder:32b", "qwen2.5-coder:7b"]


def test_prompt_input_ablation_masks_the_actual_api_context():
    signature = "(input_secret, axis_secret=0)"
    doc = "DOCUMENTATION_SECRET"
    full = build_prompt("fake.api", signature, doc, [])
    doc_only = build_prompt("fake.api", signature, doc, [], "doc-only")
    sig_only = build_prompt("fake.api", signature, doc, [], "sig-only")
    assert signature in full and doc in full
    assert signature not in doc_only and doc in doc_only
    assert doc not in sig_only and signature in sig_only


@pytest.mark.parametrize("library", LIBRARIES)
def test_library_adapters_are_explicit_and_lazy(library):
    adapter = get_adapter(library)
    assert adapter.root == library
    assert callable(adapter.documentation)
    assert callable(adapter.array)


@pytest.mark.parametrize("library", ["numpy", "scipy", "cv2", "torch"])
def test_installed_native_adapters_preserve_shape_and_dtype(library):
    pytest.importorskip(library)
    result = get_adapter(library).array(np.zeros((2, 3), dtype="float64"), "float64")
    assert tuple(result.shape) == (2, 3)
    assert "float64" in str(result.dtype)


@pytest.mark.parametrize("variant,disabled", [("noT", "type"), ("noS", "size"),
                                               ("noNF", "nonfinite"), ("noRand", "ordinary")])
def test_disabled_generation_strategies_do_not_enter_the_variant_pool(variant, disabled):
    spec = ParamSpec(name="image", kind="array")
    variants = generation.derive_variants(spec, {}, generation.VARIANTS[variant])
    assert variants
    assert all(v["strategy"] != disabled and v.get("pattern") != disabled for v in variants)


def test_nonfinite_image_pattern_is_not_lost_after_image_augmentations():
    spec = ParamSpec(name="image", kind="array")
    variants = generation.derive_variants(spec, {})
    assert any(v.get("pattern") == "nonfinite" for v in variants)
    assert any(v.get("pattern") == "extreme" for v in variants)


def test_worker_python_coverage_includes_actual_target_invocations(tmp_path):
    summary = run(RunConfig(lib="numpy", out_dir=str(tmp_path), extractor="signature",
                            apis=["numpy.clip"], max_cases=3, budget_sec=2, python_coverage=True))
    assert summary["apis_covered"] == 1
    assert summary["coverage"]["python_measured"]
    assert summary["coverage"]["python_lines"] > 0
    report = json.loads((tmp_path / "coverage" / "python.json").read_text())
    assert any("numpy" in name for name in report["files"])


def test_standalone_reproducer_uses_the_same_nonfinite_oracle(tmp_path):
    finding = Finding(api="numpy.negative", oracle="nan", kind="non_finite_output",
                      order=["x"], doc="NaNs propagate.", constraints_satisfied=True,
                      args={"x": {"recipe": {"form": "literal_array", "dtype": "float64",
                                              "values": [float("nan")]}, "by_keyword": False}})
    source = reproducer.render(finding)
    assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(source).readline))
    ast.parse(source)
    path = tmp_path / "reproducer.py"
    path.write_text(source)
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "calling" in result.stdout


def test_llm_direct_uses_one_model_call_and_the_same_worker(monkeypatch, tmp_path):
    calls = []
    def complete(prompt, config):
        calls.append(prompt)
        return "import numpy as np\n\ndef generate_args(rng):\n    return [np.array([1.0, 2.0])], {}\n"
    monkeypatch.setattr("vistafuzz.llm.complete", complete)
    config = RunConfig(lib="numpy", out_dir=str(tmp_path / "run"), generator="llm-direct",
                       max_cases=3, budget_sec=1, spec_dir=str(tmp_path / "specs"))
    work = tmp_path / "work"
    work.mkdir()
    first = run_api(collect_api("numpy.negative"), config, str(work))
    second = run_api(collect_api("numpy.negative"), config, str(work))
    assert len(calls) == 1
    assert first.executed == 3
    assert second.executed == 3


def test_native_coverage_deduplicates_source_lines(tmp_path):
    from vistafuzz.coverage_support import native_lines

    path = tmp_path / "native.info"
    path.write_text("SF:/src/kernel.cc\nDA:1,3\nDA:2,0\nend_of_record\n"
                    "SF:/src/kernel.cc\nDA:1,1\nDA:3,4\nend_of_record\n")
    assert native_lines(path) == 2


def test_repeated_ablation_runs_keep_shared_extraction_and_distinct_outputs(tmp_path):
    from vistafuzz.experiments import run_repeated

    config = RunConfig(lib="numpy", out_dir=str(tmp_path), extractor="signature",
                       apis=["numpy.negative"], max_cases=2, budget_sec=0.5)
    results = run_repeated(config, repeats=2, variants=["full", "noNF"])
    assert [r["variant"] for r in results] == ["full", "noNF"]
    assert all(r["apis_covered"]["runs"] == [1, 1] for r in results)
    assert len(list((tmp_path / "specs").glob("*.json"))) == 1
    assert (tmp_path / "experiments.json").exists()


def test_benchmark_requires_a_match_for_the_specific_reported_failure(monkeypatch, tmp_path):
    from vistafuzz.benchmark import run_benchmark

    validator = tmp_path / "validator.py"
    validator.write_text("def matches(finding):\n    return finding['detail'] == 'the expected failure'\n")
    manifest = tmp_path / "bugs.json"
    manifest.write_text(json.dumps({"bugs": [
        {"id": "fixture", "api": "fixture.api", "oracle": "nan", "validator": "validator.py"},
    ]}))
    def fake_run(config):
        directory = __import__("pathlib").Path(config.out_dir)
        directory.mkdir(parents=True)
        (directory / "apis.jsonl").write_text(json.dumps({"api": "fixture.api", "executed": 1}) + "\n")
        (directory / "findings.jsonl").write_text(json.dumps({
            "api": "fixture.api", "oracle": "nan", "detail": "a different failure"}) + "\n")
    monkeypatch.setattr("vistafuzz.benchmark.run", fake_run)
    result = run_benchmark(RunConfig(lib="fixture", out_dir=str(tmp_path / "run")),
                           str(manifest), repeats=2)
    assert result["detected"] == 0
    assert result["bugs"]["fixture"]["supported_runs"] == [1, 2]


def test_fixed_issues_are_included_in_confirmed_counts(tmp_path):
    from vistafuzz.issue_stats import summarize

    path = tmp_path / "bugs.csv"
    path.write_text("Link,Library,Bug Type,Bug Status\n"
                    "a,torch,NaN/Inf,Fixed\nb,torch,Crash/Exception,Confirmed\n"
                    "c,torch,NaN/Inf,Pending\nd,torch,NaN/Inf,Others\n")
    result = summarize(path)[0]
    assert result == {"Library": "PyTorch", "Crash": 1, "NaN": 3, "Reported": 4,
                      "Confirmed": 2, "Fixed": 1, "Pending": 1, "Excluded": 1}

