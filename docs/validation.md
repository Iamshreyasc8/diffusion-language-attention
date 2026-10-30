# Validation record

Checked on 30 October 2026. Results below distinguish executed checks from
unverified pretrained inference.

| Check | Result |
| --- | --- |
| Lightweight numerical/input/export tests | 15 passed on Python 3.12.14 |
| Same tests plus real DAAM/Diffusers integration | 16 passed on Python 3.11.17, CPU |
| Installed model dependency consistency | All 110 installed packages compatible (`uv pip check`) |
| Notebook format | All five notebooks passed nbformat 5.10.4 validation |
| Python syntax | Package/tests and notebook Python cells parsed successfully; notebook shell/magic lines excluded from AST parsing |
| Main notebook's default code path | All code cells executed sequentially in one Python process; optional pretrained generation skipped |
| Jupyter kernel execution | Blocked by this host's socket/network-interface restriction; not claimed as passed |
| Credential-pattern scan | No Hugging Face or GitHub token patterns found in repository text |
| Full pretrained SD1.5 demo | Not executed in this environment |
| CUDA inference | Not executed; no CUDA device available |

## Integration test scope

`tests/test_model_smoke.py` constructs small randomly initialized text encoder,
UNet, VAE, tokenizer, and scheduler components locally. It uses the actual
StableDiffusionPipeline and DAAM 0.2.0 to run one CPU inference step, collect
word heatmaps, calculate overlap, and export image/array/JSON files. It also
checks that an existing result directory cannot be overwritten.

This validates real library integration and output handling. It does **not**
validate pretrained weights, downloaded model access, the pretrained safety
checker, image quality, full-size performance, or scientific conclusions. The
random fixture has no pretrained safety component; the real model loader
retains the pretrained pipeline's safety checker.

To reproduce on Linux with Python 3.11:

```bash
python -m pip install -r requirements-model-cpu.lock.txt
ATTENTION_MODEL_SMOKE=1 python -m unittest discover -s tests -v
```

In PowerShell, set `$env:ATTENTION_MODEL_SMOKE="1"` before the test command.
The supplied CPU lock snapshot is Linux-specific; resolve `requirements-model.txt`
with appropriate PyTorch wheels on other platforms.

## Dependency profiles

- `requirements.txt`: pinned direct dependencies for model-independent analysis.
- `requirements-notebook.txt`: notebook UI/format support.
- `requirements-model.txt`: shared pinned core model stack, following DAAM's upstream versions.
- `requirements-model-cpu.lock.txt`: installed CPU environment that passed integration tests; includes the CPU PyTorch index.
- `requirements-model.lock.txt`: resolved Linux/Python 3.11 CUDA dependency snapshot. Dependency resolution succeeded, but installing this larger profile hit the host's cache-space limit. It was not runtime-tested. The CPU environment was installed and tested instead.

Warnings from legacy dependencies (including Matplotlib/Pyparsing and Diffusers)
were observed during passing tests. Updating this older stack should be a
separate compatibility exercise; do not upgrade individual packages blindly.

## Remaining acceptance check

Run the README's real fire-truck example in the intended environment using a
specific model commit SHA. Retain `scores.json`, raw arrays, the generated
image, and attention figure. Check that both requested word maps are present,
then document device and package versions before claiming a reproduced
pretrained-model result.
