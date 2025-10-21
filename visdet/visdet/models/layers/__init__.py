# ruff: noqa
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from visengine.model import BaseModule
from visengine.utils import to_2tuple
from .bbox_nms import multiclass_nms
from . import normed_predictor  # This registers Linear in MODELS registry


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding="corner"):
        super().__init__()
        assert padding in ("same", "corner")

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == "same":
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_channels=3,
        embed_dims=96,
        norm_cfg=None,
        conv_type="Conv2d",
        kernel_size=None,
        stride=None,
        padding="corner",
        dilation=1,
        bias=True,
        input_size=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        # Handle different parameter names
        if kernel_size is not None:
            patch_size = kernel_size
        if stride is None:
            stride = patch_size

        self.img_size = img_size
        self.patch_size = patch_size

        kernel_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)

        self.proj = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        if norm_cfg is not None:
            self.norm = nn.LayerNorm(embed_dims)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.proj(x)  # B, embed_dims, H/patch_size, W/patch_size
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B, H*W/patch_size^2, embed_dims
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)


class PatchMerging(BaseModule):
    """Patch Merging Layer."""

    def __init__(self, in_channels, out_channels, stride=2, norm_cfg=dict(type="LN"), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduction = nn.Linear(stride * stride * in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(stride * stride * in_channels)

    def forward(self, x, hw_shape):
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L} != {H} * {W}"

        # Pad if necessary to make H and W divisible by stride
        pad_h = (self.stride - H % self.stride) % self.stride
        pad_w = (self.stride - W % self.stride) % self.stride

        x = x.view(B, H, W, C)

        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w

        # Merge patches based on stride
        if self.stride == 2:
            # Extract 2x2 patches
            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C - top-left
            x1 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C - top-right
            x2 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C - bottom-left
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C - bottom-right

            # Stack to get B, H/2, W/2, 4, C
            x = torch.stack([x0, x1, x2, x3], dim=3)
            B, H_new, W_new, _, C = x.shape

            # Transpose to B, H/2, W/2, C, 4 to group by channels
            x = x.transpose(3, 4)

            # Reshape to B, H/2, W/2, 4*C with channel-major ordering
            # This matches nn.Unfold which extracts: [ch0_tl, ch0_tr, ch0_bl, ch0_br, ch1_tl, ...]
            x = x.reshape(B, H_new, W_new, 4 * C)
        else:
            # General case for any stride
            patches = []
            # Extract patches in raster scan order (top-to-bottom, left-to-right)
            for i in range(self.stride):
                for j in range(self.stride):
                    patches.append(x[:, i :: self.stride, j :: self.stride, :])

            # Stack and reshape with channel-major ordering
            x = torch.stack(patches, dim=3)  # B, H/s, W/s, s*s, C
            B, H_new, W_new, _, C = x.shape

            # Transpose to B, H/s, W/s, C, s*s to group by channels
            x = x.transpose(3, 4)

            # Reshape to match nn.Unfold ordering
            x = x.reshape(B, H_new, W_new, self.stride * self.stride * C)

        x = x.view(B, -1, self.stride * self.stride * C)

        x = self.norm(x)
        x = self.reduction(x)

        Hnew = H // self.stride
        Wnew = W // self.stride

        return x, (Hnew, Wnew)


__all__ = ["AdaptivePadding", "PatchEmbed", "PatchMerging", "multiclass_nms"]
