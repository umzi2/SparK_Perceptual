# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn


class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125
    ) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class ConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        token_mixer=nn.Identity,
        norm_layer=nn.BatchNorm2d,
        mlp_layer=ConvMlp,
        mlp_ratio=4,
        act_layer=nn.GELU,
        ls_init_value=1e-6,
        drop_path=0.0,
    ) -> None:
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        )

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = x + shortcut
        return x


class MetaNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        ds_stride=2,
        depth=2,
        drop_path_rates=None,
        ls_init_value=1.0,
        token_mixer=InceptionDWConv2d,
        act_layer=nn.GELU,
        norm_layer=None,
        mlp_ratio=4,
    ) -> None:
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                MetaNeXtBlock(
                    dim=out_chs,
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    token_mixer=token_mixer,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
            )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class MetaNeXt(nn.Module):
    r"""MetaNeXt
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt`  - https://arxiv.org/pdf/2203.xxxxx.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        token_mixers=InceptionDWConv2d,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
        mlp_ratios=(4, 4, 4, 3),
        drop_rate=0.0,
        drop_path_rate=0.0,
        ls_init_value=1e-6,
        **kwargs,
    ) -> None:
        super().__init__()

        num_stage = len(depths)
        if not isinstance(token_mixers, list | tuple):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, list | tuple):
            mlp_ratios = [mlp_ratios] * num_stage

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), norm_layer(dims[0])
        )

        self.stages = nn.Sequential()
        dp_rates = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        stages = []
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(
                MetaNeXtStage(
                    prev_chs,
                    out_chs,
                    ds_stride=2 if i > 0 else 1,
                    depth=depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    token_mixer=token_mixers[i],
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratios[i],
                )
            )
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        # self.head = head_fn(self.num_features, num_classes, drop=drop_rate)
        # self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),requires_grad=False)
        # self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1), requires_grad=False)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True) -> None:
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"norm"}

    def forward_features(self, x):
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, -1, 1, 1)
        x = self.stem(x)
        x = self.stages(x)
        return x

    def features_extract(self, x):
        f = []
        x = self.stem(x)
        for i in range(4):
            x = self.stages[i](x)
            f.append(x)
        return f

    def forward(self, x):
        return self.features_extract(x)


def inceptionnext_tiny(path: str | None = None):
    if path:
        state = torch.load(path, map_location="cpu")
    else:
        state = torch.hub.load_state_dict_from_url(
            url="https://github.com/umzi2/SparK_Perceptual/releases/download/pretrain/epoch290.pth",
            map_location="cpu",
            weights_only=True,
        )
    model = MetaNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    print(model.load_state_dict(state))
    return model.eval()
