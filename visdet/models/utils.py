# ruff: noqa
# type: ignore
import torch
from functools import partial


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(list(map_result) for map_result in zip(*map_results))


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by level."""
    target_list = []
    for level in range(num_levels):
        target_list.append([])
        for img_id in range(len(target)):
            target_list[level].append(target[img_id][level])
    return target_list


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items."""
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_shape = (count,) + data.shape[1:]
        ret = data.new_full(new_shape, fill)
        ret[inds, :] = data
    return ret


__all__ = ["multi_apply", "images_to_levels", "unmap"]
