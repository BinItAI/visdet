# Copyright (c) OpenMMLab. All rights reserved.
"""Async benchmark utility.

Note: this script is not part of the unit test suite; it is a runnable example.
It now points at a YAML preset config instead of a removed Python config.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import urllib

import torch

import visdet.cv as mmcv
from visdet.apis import async_inference_detector, inference_detector, init_detector
from visdet.utils.contextmanagers import concurrent
from visdet.utils.profiling import profile_time


async def main():
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    project_dir = os.path.join(project_dir, "..")

    config_file = os.path.join(project_dir, "configs/presets/models/mask_rcnn_r50.yaml")
    checkpoint_file = os.path.join(project_dir, "checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth")

    if not os.path.exists(checkpoint_file):
        url = (
            "https://download.openmmlab.com/mmdetection/v2.0"
            "/mask_rcnn/mask_rcnn_r50_fpn_1x_coco"
            "/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
        )
        local_filename, _ = urllib.request.urlretrieve(url)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        shutil.move(local_filename, checkpoint_file)

    device = "cuda:0"
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    streamqueue = asyncio.Queue()
    streamqueue_size = 4

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    img = mmcv.imread(os.path.join(project_dir, "demo/demo.jpg"))

    await async_inference_detector(model, img)

    async def detect(img):
        async with concurrent(streamqueue):
            return await async_inference_detector(model, img)

    num_of_images = 20
    with profile_time("benchmark", "async"):
        tasks = [asyncio.create_task(detect(img)) for _ in range(num_of_images)]
        await asyncio.gather(*tasks)

    with torch.cuda.stream(torch.cuda.default_stream()):
        with profile_time("benchmark", "sync"):
            _ = [inference_detector(model, img) for _ in range(num_of_images)]


if __name__ == "__main__":
    asyncio.run(main())
