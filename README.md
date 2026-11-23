# VistaFuzz

## Install

Use Python 3.9 or newer. Install each target library in the Python environment used to run VistaFuzz.

```bash
git clone https://github.com/beanduan22/Vistafuzz-.git
cd Vistafuzz-
python -m pip install -e ".[coverage]"
python -m pip install numpy scipy
```

Install [Ollama](https://ollama.com/download), then start its server in a separate terminal:

```bash
ollama serve
```

Download the model and configure VistaFuzz:

```bash
ollama pull qwen2.5-coder:32b
export VISTAFUZZ_LLM_BACKEND=ollama
export VISTAFUZZ_LLM_MODEL=qwen2.5-coder:32b
export VISTAFUZZ_OLLAMA_HOST=http://localhost:11434
export VISTAFUZZ_LLM_TIMEOUT=300
```

## Run

Discover APIs and extract specifications:

```bash
python -m vistafuzz.cli apis --lib numpy --limit 20
python -m vistafuzz.cli extract --api numpy.clip --spec-dir specs
```

Fuzz a library or an individual API:

```bash
python -m vistafuzz.cli fuzz --lib numpy --out runs/numpy --spec-dir specs --budget 60
python -m vistafuzz.cli fuzz --api numpy.clip --out runs/clip --spec-dir specs --budget 60
python -m vistafuzz.cli report --out runs/numpy
```

Use signatures supplied in binding documentation:

```bash
python -m vistafuzz.cli fuzz --api numpy.log --signature-policy documented --out runs/log --spec-dir specs --budget 60
```

Run repeated campaigns with coverage or component variants:

```bash
python -m vistafuzz.cli fuzz --lib numpy --out runs/repeated --spec-dir specs --budget 60 --repeat 5 --coverage
python -m vistafuzz.cli fuzz --lib numpy --out runs/variants --spec-dir specs --budget 60 --repeat 5 --variants full,noT,noS,noRand,noBound,noNF,noRel,noVal --coverage
```
