# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from collections.abc import Sized
from typing import TYPE_CHECKING, Any, Union, overload

import numpy as np
import torch

from visdet.engine.device import get_device
from visdet.engine.structures.base_data_element import BaseDataElement

if TYPE_CHECKING:
    from visdet.structures.bbox import BaseBoxes
    from visdet.structures.mask import BitmapMasks, PolygonMasks

BoolTypeTensor: type[torch.Tensor]
LongTypeTensor: type[torch.Tensor]

if get_device() == "npu":
    BoolTypeTensor = Union[torch.BoolTensor, torch.npu.BoolTensor]  # type: ignore[misc,assignment,name-defined]
    LongTypeTensor = Union[torch.LongTensor, torch.npu.LongTensor]  # type: ignore[misc,assignment,name-defined]
elif get_device() == "mlu":
    BoolTypeTensor = Union[torch.BoolTensor, torch.mlu.BoolTensor]  # type: ignore[misc,assignment,name-defined]
    LongTypeTensor = Union[torch.LongTensor, torch.mlu.LongTensor]  # type: ignore[misc,assignment,name-defined]
elif get_device() == "musa":
    BoolTypeTensor = Union[torch.BoolTensor, torch.musa.BoolTensor]  # type: ignore[misc,assignment,name-defined]
    LongTypeTensor = Union[torch.LongTensor, torch.musa.LongTensor]  # type: ignore[misc,assignment,name-defined]
else:
    BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]  # type: ignore[misc,assignment,name-defined]
    LongTypeTensor = Union[torch.LongTensor, torch.cuda.LongTensor]  # type: ignore[misc,assignment,name-defined]

IndexType = Union[str, slice, int, list[int], torch.Tensor, np.ndarray]


# Modified from
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/data_structures/instance_data.py
class InstanceData(BaseDataElement):
    """Data structure for instance-level annotations or predictions.

    Subclass of :class:`BaseDataElement`. All value in `data_fields`
    should have the same length. This design refer to
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501
    InstanceData also support extra functions: ``index``, ``slice`` and ``cat`` for data field. The type of value
    in data field can be base data structure such as `torch.Tensor`, `numpy.ndarray`, `list`, `str`, `tuple`,
    and can be customized data structure that has ``__len__``, ``__getitem__`` and ``cat`` attributes.

    Examples:
        >>> # custom data structure
        >>> class TmpObject:
        ...     def __init__(self, tmp) -> None:
        ...         assert isinstance(tmp, list)
        ...         self.tmp = tmp
        ...     def __len__(self):
        ...         return len(self.tmp)
        ...     def __getitem__(self, item):
        ...         if isinstance(item, int):
        ...             if item >= len(self) or item < -len(self):  # type:ignore
        ...                 raise IndexError(f'Index {item} out of range!')
        ...             else:
        ...                 # keep the dimension
        ...                 item = slice(item, None, len(self))
        ...         return TmpObject(self.tmp[item])
        ...     @staticmethod
        ...     def cat(tmp_objs):
        ...         assert all(isinstance(results, TmpObject) for results in tmp_objs)
        ...         if len(tmp_objs) == 1:
        ...             return tmp_objs[0]
        ...         tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        ...         tmp_list = list(itertools.chain(*tmp_list))
        ...         new_data = TmpObject(tmp_list)
        ...         return new_data
        ...     def __repr__(self):
        ...         return str(self.tmp)
        >>> from visdet.engine.structures import InstanceData
        >>> import numpy as np
        >>> import torch
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = InstanceData(metainfo=img_meta)
        >>> 'img_shape' in instance_data
        True
        >>> instance_data.det_labels = torch.LongTensor([2, 3])
        >>> instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
        >>> instance_data.bboxes = torch.rand((2, 4))
        >>> instance_data.polygons = TmpObject([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> len(instance_data)
        2
        >>> print(instance_data)
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3])
            det_scores: tensor([0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7fb492de6280>
        >>> sorted_results = instance_data[instance_data.det_scores.sort().indices]
        >>> sorted_results.det_scores
        tensor([0.7000, 0.8000])
        >>> print(instance_data[instance_data.det_scores > 0.75])
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2])
            det_scores: tensor([0.8000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188]])
            polygons: [[1, 2, 3, 4]]
        ) at 0x7f64ecf0ec40>
        >>> print(instance_data[instance_data.det_scores > 1])
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([], dtype=torch.int64)
            det_scores: tensor([])
            bboxes: tensor([], size=(0, 4))
            polygons: []
        ) at 0x7f660a6a7f70>
        >>> print(instance_data.cat([instance_data, instance_data]))
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3, 2, 3])
            det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263],
                        [0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7f203542feb0>
    """

    def __setattr__(self, name: str, value: Sized):
        """Setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f"{name} has been used as a private attribute, which is immutable.")

        else:
            assert isinstance(value, Sized), "value must contain `__len__` attribute"

            if len(self) > 0:
                assert len(value) == len(self), (
                    f"The length of values {len(value)} is not consistent with the length of this :obj:`InstanceData` {len(self)}"
                )
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> "InstanceData":
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`InstanceData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)

        if isinstance(item, str):
            return getattr(self, item)  # type: ignore[return-value]

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                raise IndexError(f"Index {item} out of range!")
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, "Only support to get the values along the first dimension."
            # Check if it's a boolean tensor
            is_bool_tensor = item.dtype == torch.bool
            if is_bool_tensor:
                assert len(item) == len(self), (
                    "The shape of the "
                    "input(BoolTensor) "
                    f"{len(item)} "
                    "does not match the shape "
                    "of the indexed tensor "
                    "in results_field "
                    f"{len(self)} at "
                    "first dimension."
                )

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, str | list | tuple) or (hasattr(v, "__getitem__") and hasattr(v, "cat")):
                    # convert to indexes from BoolTensor
                    if is_bool_tensor:
                        indexes = torch.nonzero(item).view(-1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))  # type: ignore[arg-type]
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, str | list | tuple):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)  # type: ignore[attr-defined]
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f"The type of `{k}` is `{type(v)}`, which has no attribute of `cat`, so it does not support slice with `bool`"
                    )

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data

    @staticmethod
    def cat(instances_list: list["InstanceData"]) -> "InstanceData":
        """Concat the instances of all :obj:`InstanceData` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.

        Returns:
            :obj:`InstanceData`
        """
        assert all(isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [instances.all_keys() for instances in instances_list]
        assert len({len(field_keys) for field_keys in field_keys_list}) == 1 and len(
            set(itertools.chain(*field_keys_list))
        ) == len(field_keys_list[0]), (
            "There are different keys in "
            "`instances_list`, which may "
            "cause the cat operation "
            "to fail. Please make sure all "
            "elements in `instances_list` "
            "have the exact same key."
        )

        new_data = instances_list[0].__class__(metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values: list[Any] = [results[k] for results in instances_list]
            v0 = values[0]
            new_values: Any
            # Use explicit type checking instead of isinstance to avoid mypy narrowing issues
            if type(v0).__name__ == "Tensor" or isinstance(v0, torch.Tensor):
                new_values = torch.cat(values, dim=0)  # type: ignore[arg-type]
            elif type(v0).__name__ == "ndarray" or isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)  # type: ignore[arg-type]
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v  # type: ignore[operator]
            elif hasattr(v0, "cat"):
                new_values = v0.cat(values)  # type: ignore[attr-defined]
            else:
                raise ValueError(f"The type of `{k}` is `{type(v0)}` which has no attribute of `cat`")
            new_data[k] = new_values
        return new_data

    def __len__(self) -> int:
        """int: The length of InstanceData."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0

    # Provide type hints for commonly accessed dynamic attributes
    if TYPE_CHECKING:
        # These are the most commonly accessed attributes in visualization code
        bboxes: torch.Tensor | "BaseBoxes"
        labels: torch.Tensor
        scores: torch.Tensor
        masks: torch.Tensor | "BitmapMasks" | "PolygonMasks"
        label_names: list[str]
        priors: torch.Tensor  # Used in dense heads for anchor-based detection
        level_ids: torch.Tensor  # Used to track which FPN level each instance belongs to
