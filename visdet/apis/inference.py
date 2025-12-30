# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Import models to ensure they are registered before any API function is called
from visdet import models as _  # noqa: F401

# from visdet.cv.ops import RoIPool  # Removed - eliminating C++ ops
from visdet.cv.transforms import Compose
from visdet.engine.config import Config
from visdet.engine.dataset import default_collate, pseudo_collate
from visdet.engine.model.utils import revert_sync_batchnorm
from visdet.engine.registry import init_default_scope
from visdet.engine.runner import load_checkpoint

from visdet.evaluation import get_classes
from visdet.registry import DATASETS, MODELS
from visdet.structures import DetDataSample, SampleList
from visdet.utils import ConfigType, get_test_pipeline_cfg


logger = logging.getLogger(__name__)


def _parse_inference_devices(device: str | Sequence[str] | Sequence[int]) -> tuple[str, list[int] | None]:
    if isinstance(device, str):
        if device == "cuda":
            if not torch.cuda.is_available():
                return "cpu", None

            device_count = torch.cuda.device_count()
            if device_count <= 1:
                return "cuda:0", None

            device_ids = list(range(device_count))
            return "cuda:0", device_ids

        if device.startswith("cuda") and "," in device:
            parts = [p.strip() for p in device.split(",") if p.strip()]
            device_ids = [_device_to_id(p) for p in parts]
            return f"cuda:{device_ids[0]}", device_ids

        return device, None

    device_ids = [_device_to_id(d) for d in device]
    return f"cuda:{device_ids[0]}", device_ids


def _device_to_id(device: str | int) -> int:
    if isinstance(device, int):
        return device

    if device.isdigit():
        return int(device)

    if device == "cuda":
        return 0

    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])

    raise ValueError(
        "Multi-device inference only supports CUDA devices. "
        f"Got device={device!r}; expected e.g. 'cuda:0' or an int device id"
    )


def init_detector(
    config: str | Path | Config,
    checkpoint: str | None = None,
    palette: str = "none",
    device: str | Sequence[str] | Sequence[int] = "cuda",
    cfg_options: dict | None = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`visdet.engine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str | Sequence[str] | Sequence[int]):
            The device(s) to run inference on.

            - Default: ``"cuda"`` (use all available GPUs if CUDA is available,
              otherwise fall back to CPU)
            - Single device examples: ``"cpu"``, ``"cuda:0"``
            - Multi-GPU single-process examples: ``"cuda:0,1"`` or
              ``["cuda:0", "cuda:1"]``

            When multiple CUDA devices are used, the model is wrapped in
            :class:`visdet.engine.model.wrappers.MMDataParallel`.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str | Path):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(f"config must be a filename or Config object, but got {type(config)}")
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif "init_cfg" in config.model.backbone:
        config.model.backbone.init_cfg = None

    scope = config.get("default_scope", "visdet")
    if scope is not None:
        init_default_scope(config.get("default_scope", "visdet"))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter("once")
        warnings.warn("checkpoint is None, use COCO classes by default.", stacklevel=2)
        model.dataset_meta = {"classes": get_classes("coco")}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get("meta", {})

        # save the dataset_meta in the model for convenience
        if "dataset_meta" in checkpoint_meta:
            # visdet 3.x, all keys should be lowercase
            model.dataset_meta = {k.lower(): v for k, v in checkpoint_meta["dataset_meta"].items()}
        elif "CLASSES" in checkpoint_meta:
            # < visdet 3.x
            classes = checkpoint_meta["CLASSES"]
            model.dataset_meta = {"classes": classes}
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "dataset_meta or class names are not saved in the checkpoint's meta data, use COCO classes by default.",
                stacklevel=2,
            )
            model.dataset_meta = {"classes": get_classes("coco")}

    # Priority:  args.palette -> config -> checkpoint
    if palette != "none":
        model.dataset_meta["palette"] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg["lazy_init"] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get("palette", None)
        if cfg_palette is not None:
            model.dataset_meta["palette"] = cfg_palette
        else:
            if "palette" not in model.dataset_meta:
                warnings.warn(
                    "palette does not exist, random is used by default. You can also set the palette to customize.",
                    stacklevel=2,
                )
                model.dataset_meta["palette"] = "random"

    model.cfg = config  # save the config in the model for convenience

    if device == "cuda":
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info("found %d GPUs", gpu_count)

    primary_device, device_ids = _parse_inference_devices(device)
    model.to(primary_device)

    if device_ids is not None and len(device_ids) > 1:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available, cannot use multi-GPU device={device!r}")
        if max(device_ids) >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device_ids={device_ids} but only {torch.cuda.device_count()} CUDA device(s) are available"
            )

        from visdet.engine.model import MMDataParallel

        model = MMDataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    model.eval()
    return model


ImagesType = str | np.ndarray | Sequence[str] | Sequence[np.ndarray]


def inference_detector(
    model: nn.Module,
    imgs: ImagesType,
    test_pipeline: Compose | None = None,
    text_prompt: str | None = None,
    custom_entities: bool = False,
) -> DetDataSample | SampleList:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, list | tuple):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with visdet.
            test_pipeline[0].type = "visdet.LoadImageFromNDArray"

        test_pipeline = Compose(test_pipeline)

    # RoIPool check removed - eliminating C++ ops

    data_list = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = {"img": img, "img_id": 0}
        else:
            # TODO: remove img_id.
            data_ = {"img_path": img, "img_id": 0}

        if text_prompt:
            data_["text"] = text_prompt
            data_["custom_entities"] = custom_entities

        data_list.append(test_pipeline(data_))

    batch = pseudo_collate(data_list)

    # forward the model
    with torch.inference_mode():
        results = model.test_step(batch)

    if not is_batch:
        return results[0]
    else:
        return results


# TODO: Awaiting refactoring
async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, list | tuple):
        imgs = [imgs]

    cfg = model.cfg

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = "LoadImageFromNDArray"

    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = {"img": img}
        else:
            # add information into dict
            data = {"img_info": {"filename": img}, "img_prefix": None}
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    # RoIPool check removed - eliminating C++ ops

    # We don't restore grad mode manually during concurrent inference since
    # torch.inference_mode handles version tracking for us.
    with torch.inference_mode():
        results = await model.aforward_test(data, rescale=True)
    return results


def build_test_pipeline(cfg: ConfigType) -> ConfigType:
    """Build test_pipeline for mot/vis demo. In mot/vis infer, original
    test_pipeline should remove the "LoadImageFromFile" and
    "LoadTrackAnnotations".

    Args:
         cfg (ConfigDict): The loaded config.
    Returns:
         ConfigType: new test_pipeline
    """
    # remove the "LoadImageFromFile" and "LoadTrackAnnotations" in pipeline
    transform_broadcaster = cfg.test_dataloader.dataset.pipeline[0].copy()
    for transform in transform_broadcaster["transforms"]:
        if transform["type"] == "Resize":
            transform_broadcaster["transforms"] = transform
    pack_track_inputs = cfg.test_dataloader.dataset.pipeline[-1].copy()
    test_pipeline = Compose([transform_broadcaster, pack_track_inputs])

    return test_pipeline


def inference_mot(model: nn.Module, img: np.ndarray, frame_id: int, video_len: int) -> SampleList:
    """Inference image(s) with the mot model.

    Args:
        model (nn.Module): The loaded mot model.
        img (np.ndarray): Loaded image.
        frame_id (int): frame id.
        video_len (int): demo video length
    Returns:
        SampleList: The tracking data samples.
    """
    cfg = model.cfg
    data = {
        "img": [img.astype(np.float32)],
        "frame_id": [frame_id],
        "ori_shape": [img.shape[:2]],
        "img_id": [frame_id + 1],
        "ori_video_length": [video_len],
    }

    test_pipeline = build_test_pipeline(cfg)
    data = test_pipeline(data)

    # RoIPool check removed - eliminating C++ ops

    # forward the model
    with torch.no_grad():
        data = default_collate([data])
        result = model.test_step(data)[0]
    return result


def init_track_model(
    config: str | Config,
    checkpoint: str | None = None,
    detector: str | None = None,
    reid: str | None = None,
    device: str = "cuda:0",
    cfg_options: dict | None = None,
) -> nn.Module:
    """Initialize a model from config file.

    Args:
        config (str or :obj:`visdet.engine.Config`): Config file path or the config
            object.
        checkpoint (Optional[str], optional): Checkpoint path. Defaults to
            None.
        detector (Optional[str], optional): Detector Checkpoint path, use in
            some tracking algorithms like sort.  Defaults to None.
        reid (Optional[str], optional): Reid checkpoint path. use in
            some tracking algorithms like sort. Defaults to None.
        device (str, optional): The device that the model inferences on.
            Defaults to `cuda:0`.
        cfg_options (Optional[dict], optional): Options to override some
            settings in the used config. Defaults to None.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(f"config must be a filename or Config object, but got {type(config)}")
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get("meta", {})
        # save the dataset_meta in the model for convenience
        if "dataset_meta" in checkpoint_meta:
            if "CLASSES" in checkpoint_meta["dataset_meta"]:
                value = checkpoint_meta["dataset_meta"].pop("CLASSES")
                checkpoint_meta["dataset_meta"]["classes"] = value
            model.dataset_meta = checkpoint_meta["dataset_meta"]

    if detector is not None:
        assert not (checkpoint and detector), "Error: checkpoint and detector checkpoint cannot both exist"
        load_checkpoint(model.detector, detector, map_location="cpu")

    if reid is not None:
        assert not (checkpoint and reid), "Error: checkpoint and reid checkpoint cannot both exist"
        load_checkpoint(model.reid, reid, map_location="cpu")

    # Some methods don't load checkpoints or checkpoints don't contain
    # 'dataset_meta'
    # VIS need dataset_meta, MOT don't need dataset_meta
    if not hasattr(model, "dataset_meta"):
        warnings.warn("dataset_meta or class names are missed, use None by default.", stacklevel=2)
        model.dataset_meta = {"classes": None}

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model
