# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from visdet.cv.cnn import build_norm_layer
from visdet.cv.cnn.bricks import build_activation_layer
from visdet.engine.model import BaseModule
from visdet.registry import MODELS


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for residual networks."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


@MODELS.register_module()
class LN2d(nn.Module):
    """LayerNorm for 2D tensors (channels dimension).

    Performs pointwise mean and variance normalization over the channel
    dimension for inputs with shape (batch_size, channels, height, width).

    Args:
        normalized_shape (int): Number of channels to normalize.
        eps (float): Small epsilon for numerical stability. Default: 1e-6
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_abs_pos(abs_pos, has_cls_token, hw):
    """Get absolute positional embedding resized to match input size.

    Args:
        abs_pos (Tensor): absolute positional embedding.
        has_cls_token (bool): whether model has class token.
        hw (tuple): target (height, width) for resizing.

    Returns:
        Tensor: resized positional embedding.
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


def get_rel_pos(q_size, k_size, rel_pos):
    """Get relative positional embeddings for scaled positions.

    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Tensor: extracted relative positional embeddings.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """Add decomposed relative positional embeddings to attention.

    Args:
        attn (Tensor): attention map.
        q (Tensor): query q with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings for height (Lh, C).
        rel_pos_w (Tensor): relative position embeddings for width (Lw, C).
        q_size (tuple): spatial sequence size of query (q_h, q_w).
        k_size (tuple): spatial sequence size of key (k_h, k_w).

    Returns:
        Tensor: attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    return attn


def window_partition(x, window_size):
    """Partition input into non-overlapping windows.

    Args:
        x (Tensor): input tensor with shape (B, H, W, C).
        window_size (int): window size.

    Returns:
        tuple: (windows, pad_hw) where windows has shape
            (num_windows*B, window_size, window_size, C) and pad_hw is
            the padded height and width.
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """Reverse the window partition operation.

    Args:
        windows (Tensor): windows with shape (num_windows*B, window_size, window_size, C).
        window_size (int): window size.
        pad_hw (tuple): padded (height, width).
        hw (tuple): original (height, width).

    Returns:
        Tensor: output tensor with shape (B, H, W, C).
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Attention(nn.Module):
    """Multi-head attention module with optional relative positional embeddings.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool): If True, add a learnable bias to q, k, v. Default: True
        use_rel_pos (bool): Whether to use relative position embeddings. Default: False
        rel_pos_zero_init (bool): If True, zero initialize relative position embeddings.
            Default: True
        input_size (tuple): Input spatial size (height, width) for position embeddings.
            Default: None
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # Initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Default: None
        out_features (int, optional): Number of output features. Default: None
        act_cfg (dict): Config dict for activation. Default: dict(type='GELU')
        bias (bool): Whether to use bias. Default: True
        drop (float): Dropout probability. Default: 0.0
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_cfg=dict(type="GELU"),
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """Transformer block with multi-head attention and MLP.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of hidden dimension to input dimension. Default: 4.0
        qkv_bias (bool): If True, add bias to qkv. Default: True
        drop_path (float): Stochastic depth rate. Default: 0.0
        norm_cfg (dict): Config dict for normalization. Default: dict(type='LN', eps=1e-6)
        act_cfg (dict): Config dict for activation. Default: dict(type='GELU')
        use_rel_pos (bool): Whether to use relative position embeddings. Default: False
        rel_pos_zero_init (bool): Zero initialize relative position embeddings. Default: True
        window_size (int): Window size for windowed attention. 0 = global. Default: 0
        input_size (tuple): Input spatial size (height, width). Default: None
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_cfg=dict(type="LN", eps=1e-6),
        act_cfg=dict(type="GELU"),
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
    ):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_cfg=act_cfg)

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        kernel_size (tuple): Kernel size of projection layer. Default: (16, 16)
        stride (tuple): Stride of projection layer. Default: (16, 16)
        padding (tuple): Padding of projection layer. Default: (0, 0)
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Dimension of patch embedding. Default: 768
    """

    def __init__(
        self,
        kernel_size=(16, 16),
        stride=(16, 16),
        padding=(0, 0),
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


@MODELS.register_module()
class ViT(BaseModule):
    """Vision Transformer backbone for detection.

    Pure PyTorch implementation of Vision Transformer with support for
    absolute and relative positional embeddings, windowed attention, and
    flexible patch embedding.

    Args:
        img_size (int): Input image size. Default: 1024
        patch_size (int): Patch size. Default: 16
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 768
        depth (int): Number of transformer blocks. Default: 12
        num_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): Ratio of mlp hidden dimension. Default: 4.0
        qkv_bias (bool): If True, add bias to qkv. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        norm_cfg (dict): Config for normalization. Default: dict(type='LN', eps=1e-6)
        act_cfg (dict): Config for activation. Default: dict(type='GELU')
        use_abs_pos (bool): Whether to use absolute position embeddings. Default: True
        use_rel_pos (bool): Whether to use relative position embeddings. Default: False
        rel_pos_zero_init (bool): Zero initialize relative position embeddings. Default: True
        window_size (int): Window size for windowed attention. 0 = global. Default: 0
        window_block_indexes (tuple): Block indexes to apply windowed attention. Default: (0, 1, 3, 4, 6, 7, 9, 10)
        pretrain_img_size (int): Image size during pretraining. Default: 224
        pretrain_use_cls_token (bool): Whether pretraining used class token. Default: True
        init_cfg (dict, optional): Initialization config dict. Default: None
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_cfg=dict(type="LN", eps=1e-6),
        act_cfg=dict(type="GELU"),
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=window_size if i in window_block_indexes else 0,
                    input_size=(img_size // patch_size, img_size // patch_size),
                )
                for i in range(depth)
            ]
        )

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        """Initialize weights for linear layers and layer norms."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        """Initialize weights. Called by mmengine during model initialization."""
        if self.init_cfg is None:
            self.apply(self._init_weights)

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input image tensor with shape (B, 3, H, W).

        Returns:
            Tensor: Output feature map with shape (B, C, H', W').
        """
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2]))

        for blk in self.blocks:
            x = blk(x)

        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2)

        return x
