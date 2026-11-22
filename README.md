# VistaFuzz

Implementation of **Harnessing LLMs for Document-Guided Fuzzing of Python Libraries**.

VistaFuzz collects API documentation and signatures, extracts standardized parameter specifications with a locally served LLM, validates calling information, generates arguments under individual constraints and supported relationships, materializes library-native inputs, and applies crash and NaN oracles.

The ParamSpec schema, extraction prompt, relationship resolver, generation strategies, and oracles are shared across libraries. Library-facing collectors and materialization backends are in `vistafuzz/adapters/`.

## Installation and quickstart

~~~bash
git clone https://github.com/beanduan22/Vistafuzz-.git
cd Vistafuzz-
python -m pip install -e ".[dev,coverage]"
ollama pull qwen2.5-coder:32b
export VISTAFUZZ_LLM_BACKEND=ollama
export VISTAFUZZ_LLM_MODEL=qwen2.5-coder:32b
export VISTAFUZZ_OLLAMA_HOST=http://localhost:11434
export VISTAFUZZ_LLM_TIMEOUT=300
python -m vistafuzz.cli apis --lib numpy --limit 20
python -m vistafuzz.cli extract --api numpy.clip --spec-dir specs
python -m vistafuzz.cli fuzz --lib numpy --out runs/numpy --spec-dir specs --budget 60
python -m vistafuzz.cli report --out runs/numpy
~~~

Install each target library in the interpreter used to run VistaFuzz. The paper uses Python 3.13, with separate Python 3.11/3.10 environments where required by wheel availability.

The default model is `qwen2.5-coder:32b`, served through Ollama with temperature 0 and a 4,096-token context. Extraction occurs before the per-API testing budget. Specifications are cached and reused without per-test model inference. LLM failures are extraction errors; `--extractor signature` explicitly selects a separate offline diagnostic configuration.

`--api` selects individual APIs and may be repeated. `--api-list` accepts a JSON list, a JSON object containing an `apis` list, or a text file with one API path per line.

~~~bash
python -m vistafuzz.cli fuzz --api torch.special.i0 --api torch.nn.functional.gelu --out runs/torch --spec-dir specs --budget 60
~~~

## Target libraries

The versions and eligible API counts below are specified in the manuscript. `evaluation/paper.json` records the evaluation settings.

| Library | Import name | Paper version | Eligible APIs |
|---|---|---|---:|
| PyTorch | torch | 2.12.0 | 745 |
| TensorFlow | tensorflow | 2.21.0 | 1,043 |
| JAX | jax | 0.10.1 | 658 |
| Keras | keras | 3.14.0 | 197 |
| PaddlePaddle | paddle | 3.3.1 | 900 |
| OneFlow | oneflow | 0.9.0 | 213 |
| MindSpore | mindspore | 2.9.0 | 1,209 |
| Chainer | chainer | 7.8.1 | 137 |
| NumPy | numpy | 2.2.6 | 643 |
| SciPy | scipy | 1.16.3 | 1,405 |
| scikit-learn | sklearn | 1.8.0 | 238 |
| OpenCV | cv2 | 4.10.0 | 330 |
| Total | | | 7,718 |

The paper starts from 9,114 documented numerical API candidates and excludes APIs lacking usable documentation or signatures, requiring external resources or unavailable devices, lacking outputs applicable to the oracles, or requiring substantial state or setup through other APIs.

Discovery filters numerical callables. A fixed API manifest is needed to reproduce an evaluated cohort exactly. Default campaigns test individual functions. Receiver construction is available with `--methods` for separate exploratory runs.

## Specifications and generation

ParamSpec records `name`, `kind`, `required`, `default`, `position`, `call_kind`, `constraints`, `enum_values`, `dtype_candidates`, `shape`, and `doc`. Supported cross-parameter references are stored in `relationships`.

| Relationship | Execution |
|---|---|
| `shape_follows` | Follow the referenced argument's shape |
| `axis_of` | Bound an axis by the referenced argument's rank |
| `dtype_follows` | Follow the referenced argument's dtype |

Dependencies are resolved in topological order. Cycles and unresolved references exclude the specification. Unsupported relationships remain in the documentation and are not executed.

Validation aligns names, positions, required status, defaults, and calling forms with the target signature. C-extension bindings may expose calling syntax through their runtime docstrings; these records are marked `binding_doc` rather than `runtime`. APIs without usable calling information are excluded.

Generation varies types, sizes, and values. Numerical strategies include random, boundary, extreme, mathematically sensitive, and permitted non-finite values. Image-like inputs also receive noise, masking, and division. Numerical bounds are enforced in the selected dtype, sequence lengths are preserved, and fixed shapes and ranks are respected.

Synthesis is bounded to 262,144 array elements and rank 32. Unsupported essential inputs are skipped rather than changed to violate their constraints. Native backends preserve the requested shape and dtype.

Full source documentation and extraction provenance are saved with specifications. Prompt documentation is bounded by `MAX_DOC_CHARS` in `prompt.py` and the configured model context.

## Oracles and triage

The crash oracle records fatal signals, abnormal exits, timeouts, internal failures, and unexpected exceptions. Ordinary input-validation errors count as rejected invocations. Assertion failures on arguments satisfying executable constraints are retained as candidates, including the OpenCV border-assertion pattern discussed in the paper.

The NaN oracle checks numerical outputs for non-finite values unexplained by input propagation. Calls containing non-finite inputs are checked, including unexpected NaNs at infinity-valued elements. When documentation explicitly requires NaN propagation, finite or Boolean outputs that lose it are also flagged.

Findings are candidates for minimization and documentation checking. Expected non-finite results, extraction errors, duplicates, and documented behavior are removed during triage before developer reporting. Satisfying extracted constraints does not establish that the specification is complete.

## Run artifacts and metrics

~~~text
runs/numpy/
├── summary.json
├── apis.jsonl
├── findings.jsonl
├── specs/
├── reproducers/
├── coverage/
└── work/
~~~

`specs/` is replaced by `--spec-dir` when a shared cache is selected. `coverage/` is produced when coverage measurement is enabled.

| Output | Contents |
|---|---|
| `summary.json` | API reach, extraction outcomes, VGR, relationship prevalence, preprocessing time, findings, and requested coverage |
| `apis.jsonl` | Per-API accepted/rejected calls, extraction source, signature source, relationship stratum, and preprocessing time |
| `findings.jsonl` | Candidate failures and exact argument recipes |
| `specs/` | Reusable extraction records keyed by API context and model/prompt configuration |
| `work/` | Validated specifications, worker jobs, and invocation events |
| `reproducers/` | Standalone programs using the same synthesis, materialization, and oracle logic |

VGR is accepted invocations divided by accepted plus input-rejected invocations. Generation attempts that cannot construct or materialize arguments are recorded separately. Covered APIs have at least one accepted invocation. Candidate findings are distinct from confirmed bugs.

Reproducers return 1 when a checked finding reproduces and 0 when it does not. Fatal native signals terminate the script with the corresponding non-zero process status.

## Coverage

Python coverage runs inside each API worker.

~~~bash
python -m vistafuzz.cli fuzz --lib numpy --out runs/numpy-coverage --spec-dir specs --budget 60 --coverage
~~~

Combined data and JSON line records are written under `coverage/`.

For C/C++ coverage, build the target library with GCOV instrumentation. With LCOV installed, pass `--native-build-dir` to reset and capture native counters automatically for each run. Alternatively, collect an LCOV trace for that run. Reset native counters between tools and runs. `--native-coverage path/to/native.info` imports the trace. Summaries report Python and native line counts separately and their sum. Runs without a native trace do not claim native coverage.

## Repeated experiments and ablations

Every tool receives 60 seconds per supported API. One-time preprocessing is outside the budget; generation and invocation are inside it. Comparisons require the same API cohort, instrumented build, hardware, and oracles.

~~~bash
python -m vistafuzz.cli fuzz --lib numpy --out runs/numpy-five --spec-dir specs --budget 60 --repeat 5 --coverage
python -m vistafuzz.cli fuzz --lib numpy --out runs/numpy-ablation --spec-dir specs --budget 60 --repeat 5 --variants full,noT,noS,noRand,noBound,noNF,noRel,noVal --coverage
python -m vistafuzz.cli fuzz --lib numpy --out runs/numpy-budget --spec-dir specs --repeat 5 --budgets 15,30,60,120 --coverage
~~~

| Variant | Disabled component |
|---|---|
| `noT` | Type generation |
| `noS` | Size generation |
| `noRand` | Ordinary random values and random noise |
| `noBound` | Boundary, extreme, and mathematically sensitive value pools |
| `noNF` | Non-finite generation |
| `noRel` | Relationship resolution |
| `noVal` | Runtime-signature validation |

`noRel` retains individual constraints and relationship labels for `R+`/`R-` analysis. `noVal` uses the same cached raw extraction as the full configuration.

`experiments.json` contains per-run values, means, standard deviations, and candidate failure-group unions. Paper issue counts require a triage mapping from candidates to actual issues.

The `common` command intersects per-tool `apis.jsonl` files using accepted invocations. Supply its output to every compared tool.

~~~bash
python -m vistafuzz.cli common runs/vistafuzz/apis.jsonl runs/baseline/apis.jsonl --out common-apis.json
python -m vistafuzz.cli fuzz --lib torch --api-list common-apis.json --out runs/common --spec-dir specs --budget 60 --repeat 5 --coverage
~~~

## Same-model LLM-Direct

LLM-Direct gives the same model the documentation and signature once before testing. It generates `generate_args(rng)`, returning positional and keyword arguments. The cached function runs in the same worker, budget, and oracles, without ParamSpec-based generation or relationship resolution.

~~~bash
python -m vistafuzz.cli fuzz --lib torch --api-list common-apis.json --generator llm-direct --out runs/direct --spec-dir direct-generators --budget 60 --repeat 5 --coverage
~~~

The released implementations and library-specific resources of FreeFuzz, DocTer, ACETest, TitanFuzz, and FUTURE must be installed separately.

## Prompt, model, and known-bug analyses

~~~bash
python -m vistafuzz.cli fuzz --lib numpy --api-list sampled-apis.json --out runs/doc-sig --spec-dir specs --prompt-input doc+sig
python -m vistafuzz.cli fuzz --lib numpy --api-list sampled-apis.json --out runs/doc-only --spec-dir specs --prompt-input doc-only
python -m vistafuzz.cli fuzz --lib numpy --api-list sampled-apis.json --out runs/sig-only --spec-dir specs --prompt-input sig-only
python -m vistafuzz.cli fuzz --lib numpy --api-list sampled-apis.json --out runs/model-7b --spec-dir specs --llm-model qwen2.5-coder:7b
~~~

Prompt and model configurations have independent cache identities. The paper uses 240 APIs for prompt ablation, 180 for model comparison and relationship analysis, and 360 for specification accuracy. Accuracy and agreement analyses require the original annotations.

A known-bug manifest contains a `bugs` list. Each record supplies `id`, `api`, `oracle` (`crash` or `nan`), and `validator`. A validator is a Python file exposing `matches(finding)`, or a `file.py:function` reference, which checks the specific reported failure.

~~~bash
python -m vistafuzz.cli fuzz --lib torch --benchmark known-bugs.json --out runs/known-bugs --spec-dir specs --budget 60 --repeat 5
~~~

A bug is detected if its validator matches a flagged case in any run. Unsupported APIs miss their bugs. Use actual confirmed bugs present in the evaluated library versions.

The historical API manifests, extracted specifications, five-run coverage data, annotations, and 27 benchmark records are not included in this checkout. `evaluation/paper.json` contains manuscript settings, not regenerated results. `bug_review.csv` is preserved as the issue-status ledger. `python -m vistafuzz.cli bugs --file bug_review.csv` prints its current counts in the paper's column order; Fixed is included in Confirmed.

## Tests

~~~bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
~~~
