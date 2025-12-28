# Installation

VisDet is published as a normal Python package. In most cases you **do not** need to clone this repository to use it.

Cloning the repo is only needed if you want to:

- Develop VisDet itself (editable install, tests, docs)
- Use the repo’s training scripts under `tools/`
- Use the repo’s example assets/configs as-is

## Requirements

- Python `>=3.10,<3.13`
- PyTorch + torchvision
  - For CUDA / GPU support, install PyTorch following https://pytorch.org first (so you get the correct CUDA build).

## Install with uv (recommended)

**Step 0.** Install [uv](https://docs.astral.sh/uv/) (if you don’t already have it).

```shell
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Step 1.** Create a virtual environment.

```shell
uv venv --python 3.12
```

**Step 2.** Activate it.

```shell
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Step 3.** Install VisDet.

```shell
uv pip install visdet
```

### Optional extras

```shell
# Extra (optional) dependencies used by some features
uv pip install "visdet[optional]"

# Everything in the optional group
uv pip install "visdet[all]"
```

## Install with pip

If you prefer standard tooling:

```shell
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -U pip
pip install visdet
```

## Verify the installation

A minimal smoke-check (no repo clone required):

```shell
python -c "import visdet; print(visdet.__version__)"
```

Optional: run a quick inference using a built-in YAML preset (this may download model weights on first use):

```python
from visdet.apis import DetInferencer

inferencer = DetInferencer(model="rtmdet-s", device="cpu")
results = inferencer("path/to/your_image.jpg")
print(results)
```

## Install from source (development)

Only needed if you’re contributing to VisDet.

```shell
git clone <your-repository-url>
cd visdet
uv sync
```

That will:

- Create a virtual environment
- Install dependencies from `pyproject.toml`
- Install VisDet in editable mode

## Install on Google Colab

Colab usually already has PyTorch installed.

```shell
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv venv --python 3.12
!uv pip install visdet
```

```python
import visdet
print(visdet.__version__)
```

```{note}
Within Jupyter, the exclamation mark `!` runs shell commands.
```

## Using VisDet with Docker

The repo contains a `docker/` folder with a Dockerfile. This path **does** require cloning the repository.

```shell
docker build -t visdet docker/
```

Run it with:

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/visdet/data visdet
```

## Troubleshooting

- **Import errors**: ensure you installed into the environment you’re running (`which python` / `python -V`).
- **CUDA issues**: install PyTorch for your CUDA version (then reinstall `visdet` if needed).
- **Version conflicts**: try a fresh env: `rm -rf .venv && uv venv && uv pip install visdet`.
