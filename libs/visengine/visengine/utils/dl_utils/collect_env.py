import sys
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import torch
import torchvision

from visengine.device import is_cuda_available
from visengine.version import __version__ as visengine_version


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - MSVC: Microsoft Virtual C++ Compiler version, Windows only.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV (optional): OpenCV version.
            - MMENGINE: MMENGINE version.
    """

    env_info = OrderedDict()
    env_info["sys.platform"] = sys.platform
    env_info["Python"] = sys.version.replace("\n", "")

    cuda_available = is_cuda_available()
    env_info["CUDA available"] = cuda_available
    env_info["numpy_random_seed"] = np.random.get_state()[1][0]

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info["GPU " + ",".join(device_ids)] = name

    env_info["PyTorch"] = torch.__version__
    env_info["TorchVision"] = torchvision.__version__
    env_info["OpenCV"] = cv2.__version__
    env_info["VisEngine"] = visengine_version

    return env_info
