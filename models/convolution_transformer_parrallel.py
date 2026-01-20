

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer_v2 import SwinTransformerV2
from .edsr import EDSR
import numpy as np

from models import register

class SWINConvParallel(nn.Module):
    def __init__(self, args):
        super(SWINConvParallel, self).__init__()

        self.conv_model = EDSR(args)

        self.swin_model = SwinTransformerV2(img_size=256, patch_size=1, in_chans=3, num_classes=1000,
                 embed_dim=args.n_feats, depths=args.depths, num_heads=args.num_heads,
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
        self.out_dim  = 2*args.n_feats


    def forward(self, x):
        x_conv = self.conv_model(x)
        x_tran = self.swin_model(x)
        x = torch.cat((x_conv,x_tran),dim=1)
        return x

class SWINConvParallelAdd(nn.Module):
    def __init__(self, args):
        super(SWINConvParallelAdd, self).__init__()

        self.conv_model = EDSR(args)

        self.swin_model = SwinTransformerV2(img_size=256, patch_size=1, in_chans=3, num_classes=1000,
                 embed_dim=args.n_feats, depths=args.depths, num_heads=args.num_heads,
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
        self.out_dim  = args.n_feats


    def forward(self, x):
        x_conv = self.conv_model(x)
        x_tran = self.swin_model(x)
        x = x_conv + x_tran
        return x

class SWINConvParallelAdd64(nn.Module):
    def __init__(self, args):
        super(SWINConvParallelAdd64, self).__init__()

        self.conv_model = EDSR(args)

        self.swin_model = SwinTransformerV2(img_size=64, patch_size=1, in_chans=3, num_classes=1000,
                 embed_dim=args.n_feats, depths=args.depths, num_heads=args.num_heads,
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
        self.out_dim  = args.n_feats


    def forward(self, x):
        x_conv = self.conv_model(x)
        x_tran = self.swin_model(x)
        x = x_conv + x_tran
        return x

@register('swin-conv-parallel-add-small')
def make_swinir(n_resblocks=16, n_feats=16, res_scale=0.1,
                scale=2, no_upsampling=False, rgb_range=1):

    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.depths    = [2, 2, 6, 2]
    args.num_heads = [4, 4, 4, 4]
    return SWINConvParallelAdd(args) 

@register('swin-conv-parallel-add')
def make_swinir(n_resblocks=16, n_feats=32, res_scale=0.1,
                scale=2, no_upsampling=False, rgb_range=1):

    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.depths    = [2, 6, 6, 6]
    args.num_heads = [4, 4, 4, 4]
    return SWINConvParallelAdd(args) 


@register('swin-conv-parallel-add-l')
def make_swinir(n_resblocks=16, n_feats=64, res_scale=0.1,
                scale=2, no_upsampling=False, rgb_range=1):

    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.depths    = [6, 6, 6, 6]
    args.num_heads = [8, 8, 8, 8]
    return SWINConvParallelAdd(args)

@register('swin-conv-parallel-add-l-64')
def make_swinir(n_resblocks=16, n_feats=64, res_scale=0.1,
                scale=2, no_upsampling=False, rgb_range=1):

    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.depths    = [6, 6, 6, 6]
    args.num_heads = [8, 8, 8, 8]
    return SWINConvParallelAdd64(args)
