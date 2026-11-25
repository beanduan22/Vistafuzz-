# API specs

Specs extracted by VistaFuzz for 47,445 APIs across 46 libraries, one JSON object per line, one file per library.

Each line contains `api`, `backend`, `module`, `name`, `signature`, `params`, `constraints`, `doc`, `numeric_score`, and `skip_reason`.
When an API showed up in more than one run, only the newest extraction is kept.

Read them directly:

```python
import json
specs = [json.loads(l) for l in open("specs/torch.jsonl")]
```

| library | APIs | with constraints |
|---|---:|---:|
| mindspore | 10,221 | 1,446 |
| paddle | 4,300 | 370 |
| sympy | 4,122 | 0 |
| tensorflow | 3,521 | 487 |
| torch | 3,469 | 182 |
| mxnet | 3,460 | 892 |
| scipy | 2,912 | 328 |
| numpy | 2,702 | 858 |
| networkx | 1,601 | 4 |
| faiss | 1,222 | 0 |
| jax | 1,209 | 249 |
| keras | 933 | 165 |
| pandas | 891 | 5 |
| cv2 | 780 | 2 |
| flax | 566 | 37 |
| chainer | 548 | 64 |
| torchmetrics | 496 | 31 |
| cupy | 427 | 25 |
| oneflow | 384 | 54 |
| timm | 379 | 2 |
| optax | 366 | 36 |
| mlx | 323 | 59 |
| triton | 323 | 20 |
| torchvision | 302 | 2 |
| pyarrow | 297 | 0 |
| diffusers | 209 | 2 |
| torch_fidelity | 169 | 0 |
| onnx2torch | 156 | 0 |
| sklearn | 150 | 0 |
| open_clip | 146 | 0 |
| onnx | 120 | 0 |
| torchaudio | 118 | 6 |
| transformers | 115 | 2 |
| tinygrad | 92 | 1 |
| onnx2pytorch | 74 | 0 |
| onnxscript | 58 | 0 |
| math | 57 | 0 |
| onnxruntime | 57 | 0 |
| PIL | 42 | 0 |
| statistics | 34 | 0 |
| cupyx | 32 | 2 |
| cmath | 23 | 0 |
| einops | 20 | 2 |
| safetensors | 11 | 0 |
| sentencepiece | 5 | 0 |
| tokenizers | 3 | 0 |
