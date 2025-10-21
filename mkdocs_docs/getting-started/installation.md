# Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMDetection works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.5+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Install [uv](https://docs.astral.sh/uv/) - a fast Python package manager.

```shell
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Step 1.** uv will automatically manage Python versions and virtual environments for you. No separate setup needed!

# Installation

We recommend that users follow our best practices to install MMDetection using uv. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

## Best Practices

**Step 0.** Clone the repository and navigate to it.

```shell
git clone https://github.com/BinItAI/visdet.git
cd visdet
```

**Step 1.** Install dependencies using uv.

For development (includes all dependencies):

```shell
uv sync
```

For specific extras (e.g., just documentation):

```shell
uv sync --extra mkdocs
```

**Step 2.** That's it! uv has:
- Created a virtual environment
- Installed Python 3.12 (or the version specified in `.python-version`)
- Installed all dependencies from `pyproject.toml`
- Installed the package in editable mode

## Verify the installation

To verify whether MMDetection is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
uv run mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `yolov3_mobilenetv2_320_300e_coco.py` and `yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth` in your current folder.

**Step 2.** Verify the inference demo.

```shell
uv run python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```

You will see a new image `result.jpg` on your current folder, where bounding boxes are plotted on cars, benches, etc.

Alternatively, you can run Python code directly:

```python
# Run with: uv run python
from mmdet.apis import init_detector, inference_detector

config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

You will see a list of arrays printed, indicating the detected bounding boxes.

## Customize Installation

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
uv handles PyTorch installation automatically based on your `pyproject.toml` configuration. For specific CUDA versions, you may need to configure the PyTorch index URL in your project settings.
```

### Installing additional packages

To add additional packages to your project:

```shell
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add an optional dependency to a specific group
uv add --optional group-name package-name
```

### Install on CPU-only platforms

MMDetection can be built for CPU only environment. In CPU mode you can train (requires MMCV version >= 1.4.4), test or inference a model.

However some functionalities are gone in this mode:

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- Deformable ROI pooling
- CARAFE
- SyncBatchNorm
- CrissCrossAttention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps

If you try to train/test/inference a model containing above ops, an error will be raised.
The following table lists affected algorithms.

|                        Operator                         |                                          Model                                           |
| :-----------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN、Guided Anchoring、RepPoints、CentripetalNet、VFNet、CascadeRPN、NAS-FCOS、DetectoRS |
|                      MaskedConv2d                       |                                     Guided Anchoring                                     |
|                         CARAFE                          |                                          CARAFE                                          |
|                      SyncBatchNorm                      |                                         ResNeSt                                          |

### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed.
Here's how to install visdet with uv on Colab:

**Step 1.** Install uv in Colab.

```shell
!curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2.** Clone and install visdet.

```shell
!git clone https://github.com/BinItAI/visdet.git
%cd visdet
!uv sync
```

**Step 3.** Verification.

```python
!uv run python -c "import mmdet; print(mmdet.__version__)"
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

### Using MMDetection with Docker

We provide a [Dockerfile](https://github.com/BinItAI/visdet/blob/master/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmdetection docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

### Running commands with uv

All commands should be prefixed with `uv run` to ensure they use the project's virtual environment:

```shell
# Running Python scripts
uv run python your_script.py

# Running installed CLI tools
uv run pytest tests/

# Running pre-commit hooks
uv run pre-commit run --all-files

# Building documentation
uv run mkdocs build
```

Alternatively, you can activate the virtual environment manually:

```shell
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Now you can run commands without 'uv run' prefix
python your_script.py
pytest tests/
```

## Trouble shooting

If you have some issues during the installation, please first view the [FAQ](faq.md) page.
You may [open an issue](https://github.com/BinItAI/visdet/issues/new/choose) on GitHub if no solution is found.
