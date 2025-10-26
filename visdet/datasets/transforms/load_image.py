# Copyright (c) OpenMMLab. All rights reserved.

import warnings

import numpy as np

import visdet.cv.fileio as fileio
from visdet.cv import imfrombytes
from visdet.cv.transforms import BaseTransform
from visdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`visdet.cv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`visdet.cv.imfrombytes`.
            See :func:`visdet.cv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        imdecode_backend: str = "cv2",
        file_client_args: dict | None = None,
        ignore_empty: bool = False,
        *,
        backend_args: dict | None = None,
    ) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: dict | None = None
        self.backend_args: dict | None = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. Please use "backend_args" instead',
                DeprecationWarning,
            )
            if backend_args is not None:
                raise ValueError('"file_client_args" and "backend_args" cannot be set at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> dict | None:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        # Support both new img_path API and legacy img_prefix + img_info API
        if "img_path" in results:
            filename = results["img_path"]
        elif "img_prefix" in results and "img_info" in results:
            # Legacy API: img_prefix + img_info
            import os.path as osp

            img_prefix = results["img_prefix"]
            img_info = results["img_info"]
            filename = osp.join(img_prefix, img_info["filename"])
        else:
            raise KeyError("Either 'img_path' or ('img_prefix' + 'img_info') must be provided")
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(filename, backend_args=self.backend_args)

            img = imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f"failed to load image: {filename}"
        if self.to_float32:
            img = img.astype(np.float32)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"ignore_empty={self.ignore_empty}, "
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"imdecode_backend='{self.imdecode_backend}', "
        )

        if self.file_client_args is not None:
            repr_str += f"file_client_args={self.file_client_args})"
        else:
            repr_str += f"backend_args={self.backend_args})"

        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def transform(self, results: dict) -> dict | None:
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results["img"]
        if self.to_float32:
            img = img.astype(np.float32)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadMultiChannelImageFromFiles(BaseTransform):
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`visdet.cv.imfrombytes`.
            Defaults to 'unchanged'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "unchanged",
        file_client_args: dict | None = None,
    ) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy() if file_client_args is not None else None

    def transform(self, results: dict) -> dict | None:
        """Load multiple images and get images meta information.

        Args:
            results (dict): Result dict from dataset.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        img_paths = results.get("img_path", [])
        if isinstance(img_paths, str):
            img_paths = [img_paths]

        img_list = []
        for img_path in img_paths:
            try:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(self.file_client_args, img_path)
                    img_bytes = file_client.get(img_path)
                else:
                    img_bytes = fileio.get(img_path)

                img = imfrombytes(img_bytes, flag=self.color_type)
                img_list.append(img)
            except Exception as e:
                raise e

        img = np.stack(img_list, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"file_client_args={self.file_client_args})"
        )
        return repr_str
