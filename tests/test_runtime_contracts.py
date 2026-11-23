from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vistafuzz import oracles, reproducer
from vistafuzz.native import get_runtime
from vistafuzz.native_base import Runtime
from vistafuzz.collector import collect_api, has_runtime_signature, signature_params
from vistafuzz.extraction import extract
from vistafuzz.models import Finding
from vistafuzz.runner import RunConfig, run
from vistafuzz.triage import review_findings
from vistafuzz.validation import validate


def test_binding_doc_does_not_pass_independent_runtime_validation():
    record = collect_api("numpy.log")
    assert record.signature_source == "binding_doc"
    assert not has_runtime_signature(record)
    spec = extract(record, extractor="signature")
    result = validate(spec, record)
    assert not result.ok and "runtime signature" in result.reason
    assert validate(spec, record, signature_policy="documented").ok


def test_missing_runtime_signature_is_excluded_before_model_inference(monkeypatch, tmp_path):
    def unexpected_inference(*args, **kwargs):
        pytest.fail("excluded APIs must not invoke the extraction model")
    monkeypatch.setattr("vistafuzz.llm.complete", unexpected_inference)
    summary = run(RunConfig(lib="numpy", out_dir=str(tmp_path), apis=["numpy.log"]))
    assert summary["apis_excluded"] == 1
    assert summary["cases_generated"] == 0
    assert summary["signature_policy"] == "runtime"


def test_runtime_native_schema_is_used_without_a_docstring_signature():
    torch = pytest.importorskip("torch")
    record = collect_api("torch.ops.aten.special_i0.default")
    assert has_runtime_signature(record)
    assert record.signature_source == "runtime"
    assert [row["name"] for row in signature_params(record)] == ["self"]
    record.signature.bind(torch.tensor([0.0]))
    with pytest.raises(TypeError):
        record.signature.bind()


def test_documented_log_zero_is_not_a_nan_oracle_failure():
    with np.errstate(divide="ignore"):
        output = np.log(np.array([0.0, 1.0]))
    assert oracles.check_nan(output, [np.array([0.0, 1.0])], np.log.__doc__, api="numpy.log") == (False, "")
    assert oracles.check_nan(np.array([np.inf]), [np.array([0.0])], np.log.__doc__, api="numpy.log")[0]
    assert oracles.check_nan(np.array([0.0]), [np.array([0.0])], np.log.__doc__, api="numpy.log")[0]


def test_document_examples_are_not_executed(tmp_path):
    marker = tmp_path / "executed"
    doc = ("Element-wise operation.\n\n>>> numerical(__import__('pathlib').Path("
           + repr(str(marker)) + ").write_text('executed'))\n-inf\n")
    assert oracles.check_nan(np.inf, [0.0], doc, api="fixture.numerical")[0]
    assert not marker.exists()


def test_documented_exception_does_not_hide_another_input_failure():
    doc = "Element-wise operation.\n\n>>> numerical(0)\n-inf\n"
    assert oracles.check_nan(-np.inf, [0.0], doc, api="fixture.numerical") == (False, "")
    assert oracles.check_nan(-np.inf, [1.0], doc, api="fixture.numerical")[0]


def test_dtype_specific_document_examples_do_not_hide_other_dtype_failures():
    doc = ("Element-wise operation.\n\n"
           ">>> numerical(np.array([1000], dtype=np.float16))\n"
           "array([inf], dtype=float16)\n")
    assert oracles.check_nan(np.array([np.inf]), [np.array([1000.0])], doc,
                             api="fixture.numerical")[0]


def test_documented_scalar_output_with_incompatible_input_shape_does_not_crash():
    doc = "Element-wise operation.\n\n>>> numerical(0)\n-inf\n"
    assert oracles.check_nan(-np.inf, [np.array([0, 1])], doc, api="fixture.numerical")[0]


@pytest.mark.parametrize("library", ["torch", "cv2", "sklearn"])
def test_runtime_handles_readonly_negative_stride_without_changing_values(library):
    pytest.importorskip(library)
    value = np.arange(6, dtype="float64").reshape(2, 3)[:, ::-1]
    value.flags.writeable = False
    out = get_runtime(library).array(value, "float64")
    observed = out.numpy() if library == "torch" else out
    np.testing.assert_array_equal(observed, value)
    assert tuple(out.shape) == value.shape
    assert str(out.dtype).endswith("float64")


def test_standalone_reproducer_retains_native_runtime_behavior(tmp_path):
    pytest.importorskip("torch")
    finding = Finding(api="torch.neg", oracle="nan", kind="non_finite_output", order=["input"],
                      args={"input": {"recipe": {"form": "literal_array", "dtype": "float64",
                                                 "values": [1.0]}, "by_keyword": False}})
    path = tmp_path / "native_reproducer.py"
    path.write_text(reproducer.render(finding))
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "TorchRuntime" in path.read_text()


def test_paddle_nonfinite_outputs_are_visible_to_the_shared_oracle():
    pytest.importorskip("paddle")
    out = get_runtime("paddle").array(np.array([np.inf], dtype="float64"), "float64")
    assert not oracles.inputs_are_finite([out])
    assert oracles.check_nan(out, [np.array([1.0])])[0]


def test_runtime_rejects_a_signedness_change():
    runtime = Runtime("fixture", "numpy.asarray")
    value = np.array([1], dtype="int32")
    with pytest.raises(ValueError, match="changed"):
        runtime.check_array(value.astype("uint32"), value, "int32")


def test_runtime_checks_inferred_dtype_when_no_dtype_was_supplied():
    runtime = Runtime("fixture", "numpy.asarray")
    value = np.array([0.1], dtype="float64")
    with pytest.raises(ValueError, match="changed"):
        runtime.check_array(value.astype("float32"), value, None)


def test_jax_preserves_inferred_float64_with_x64_initially_disabled():
    jax = pytest.importorskip("jax")
    original = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        value = np.array([0.1], dtype="float64")
        result = get_runtime("jax").array(value, None)
        np.testing.assert_array_equal(np.asarray(result), value)
        assert str(result.dtype) == "float64"
    finally:
        jax.config.update("jax_enable_x64", original)


@pytest.mark.parametrize("dtype", ["uint16", "uint32", "uint64"])
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
def test_paddle_unsigned_conversion_preserves_integer_limits(dtype, shape):
    pytest.importorskip("paddle")
    values = np.resize(np.array([0, 1, np.iinfo(dtype).max], dtype=dtype), shape)
    out = get_runtime("paddle").array(values, dtype)
    assert out.dtype == getattr(__import__("paddle"), dtype)
    np.testing.assert_array_equal(out.numpy(), values)


def test_paddle_uint16_is_integer_even_when_its_display_name_is_bfloat16():
    paddle = pytest.importorskip("paddle")
    values = np.array([0, 1, 32768, 65535], dtype="uint16")
    out = get_runtime("paddle").array(values, None)
    assert out.dtype == paddle.uint16
    assert out.dtype != paddle.bfloat16
    assert out.numpy().dtype == np.dtype("uint16")
    np.testing.assert_array_equal(out.numpy(), values)


def test_paddle_unsigned_inputs_and_reproducers(tmp_path):
    paddle = pytest.importorskip("paddle")
    for dtype in ["uint16", "uint32", "uint64"]:
        values = np.array([0, 1, np.iinfo(dtype).max], dtype=dtype)
        native = get_runtime("paddle").array(values, dtype)
        np.testing.assert_array_equal(native.numpy(), values)
        assert paddle.numel(native).item() == 3
    finding = Finding(api="paddle.numel", oracle="nan", kind="non_finite_output",
                      order=["x"], args={
                          "x": {"recipe": {"form": "literal_array", "dtype": "uint64",
                                            "values": [0, 2**63, 2**64 - 1]}, "by_keyword": False}})
    path = tmp_path / "paddle_unsigned_repro.py"
    path.write_text(reproducer.render(finding))
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_missing_paddle_unsigned_cpu_kernel_is_an_input_rejection():
    paddle = pytest.importorskip("paddle")
    values = np.array([0, 1, 65535], dtype="uint16")
    native = get_runtime("paddle").array(values, "uint16")
    try:
        out = paddle.reshape(native, [3, 1])
    except RuntimeError as exc:
        message = str(exc)
        assert "kernel" in message and "not registered" in message
        assert oracles.classify_exception("RuntimeError", message, constraints_satisfied=True) == "rejected"
    else:
        np.testing.assert_array_equal(out.numpy(), values.reshape(3, 1))


def test_keras_torch_preserves_unsigned_inputs_and_reproducers(tmp_path):
    if importlib.util.find_spec("keras") is None or importlib.util.find_spec("torch") is None:
        pytest.skip("Keras and PyTorch are required")
    code = r'''
import json
from pathlib import Path
import numpy as np
import keras
from vistafuzz.native import get_runtime
from vistafuzz.models import Finding
from vistafuzz.reproducer import render
for dtype in ['uint16', 'uint32', 'uint64']:
    values=np.array([0, 1, np.iinfo(dtype).max],dtype=dtype)
    native=get_runtime('keras').array(values,dtype)
    assert str(native.dtype)=='torch.'+dtype
    result=keras.ops.reshape(native,(3,1))
    np.testing.assert_array_equal(result.numpy(),values.reshape(3,1))
    for shape in [(),(0,3)]:
        value=np.zeros(shape,dtype=dtype)
        out=get_runtime('keras').array(value,None)
        np.testing.assert_array_equal(out.numpy(),value)
finding=Finding(api='keras.ops.reshape',oracle='nan',kind='non_finite_output',
    order=['x','shape'],args={
        'x':{'recipe':{'form':'literal_array','dtype':'uint64',
                      'values':[0,2**63,2**64-1]},'by_keyword':False},
        'shape':{'recipe':{'form':'const','value':[3,1]},'by_keyword':False}})
Path(REPRODUCER_PATH).write_text(render(finding))
'''
    path = tmp_path / "keras_unsigned_repro.py"
    env = {**os.environ, "KERAS_BACKEND": "torch", "KERAS_TORCH_DEVICE": "cpu",
           "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", "REPRODUCER_PATH=" + repr(str(path)) + "\n" + code],
                            capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0, result.stderr
    reproduction = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                                  timeout=30, env=env)
    assert reproduction.returncode == 0, reproduction.stderr


def _triage_fixture(tmp_path):
    findings = tmp_path / "findings.jsonl"
    findings.write_text(json.dumps({"api": "fixture.numerical", "oracle": "nan"}) + "\n")
    minimal = tmp_path / "minimal.py"
    minimal.write_text("raise RuntimeError('test fixture')\n")
    decision = {"finding_index": 1, "api": "fixture.numerical", "decision": "reported",
                "documented_input_valid": True, "behavior_violates_documentation": True,
                "minimized": True, "reproduced": True, "minimized_reproducer": "minimal.py",
                "documentation_evidence": "Finite outputs are required for these inputs.",
                "reason": "The minimized case contradicts the documented behavior."}
    return findings, decision


@pytest.mark.parametrize("missing", ["documented_input_valid", "behavior_violates_documentation",
                                    "minimized", "reproduced", "documentation_evidence",
                                    "minimized_reproducer"])
def test_reporting_requires_documentation_check_and_minimized_reproducer(tmp_path, missing):
    findings, decision = _triage_fixture(tmp_path)
    decision.pop(missing)
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps([decision]))
    with pytest.raises(ValueError):
        review_findings(findings, reviews, tmp_path / "triage")
    assert not (tmp_path / "triage" / "reports.jsonl").exists()


def test_only_reviewed_documentation_violations_enter_reports(tmp_path):
    findings, decision = _triage_fixture(tmp_path)
    with findings.open("a") as handle:
        handle.write(json.dumps({"api": "fixture.expected", "oracle": "nan"}) + "\n")
        handle.write(json.dumps({"api": "fixture.pending", "oracle": "crash"}) + "\n")
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps([decision, {"finding_index": 2, "api": "fixture.expected",
        "decision": "excluded", "reason": "Non-finite output is documented for this input."}]))
    summary = review_findings(findings, reviews, tmp_path / "triage")
    assert summary == {"flagged": 3, "reported": 1, "excluded": 1, "pending": 1}
    reports = (tmp_path / "triage" / "reports.jsonl").read_text().splitlines()
    assert len(reports) == 1 and json.loads(reports[0])["api"] == "fixture.numerical"
