import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from functools import partial
from torch.nn.modules.utils import _pair
from torch.nn import init

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

# from mmseg.utils import get_root_logger
from ..builder import BACKBONES
# from mmcv.runner import load_checkpoint


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HireMLP(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0., pixel=2, step=1,
                 step_pad_mode='c', pixel_pad_mode='c', norm_style=nn.BatchNorm2d):
        super().__init__()
        self.pixel = pixel
        self.step = step
        self.step_pad_mode = step_pad_mode
        self.pixel_pad_mode = pixel_pad_mode
        print('pixel: {} pad mode: {} step: {} pad mode: {}'.format(
              pixel, pixel_pad_mode, step, step_pad_mode))
        
        self.mlp_h1 = nn.Conv2d(dim*pixel, dim//2, 1, bias=False)
        self.mlp_h1_norm = norm_style(dim//2)
        self.mlp_h2 = nn.Conv2d(dim//2, dim*pixel, 1, bias=True)
        self.mlp_w1 = nn.Conv2d(dim*pixel, dim//2, 1, bias=False)
        self.mlp_w1_norm = norm_style(dim//2)
        self.mlp_w2 = nn.Conv2d(dim//2, dim*pixel, 1, bias=True)
        self.mlp_c = nn.Conv2d(dim, dim, 1, bias=True)
        
        self.act = nn.ReLU()

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """
        
        B, C, H, W = x.shape

        pad_h, pad_w = (self.pixel - H % self.pixel) % self.pixel, (self.pixel - W % self.pixel) % self.pixel
        h, w = x.clone(), x.clone()
        
        if self.step:
            if self.step_pad_mode == '0':
                h = F.pad(h, (0, 0, self.step, 0), "constant", 0)
                w = F.pad(w, (self.step, 0, 0, 0), "constant", 0)
            elif self.step_pad_mode == 'c':
                h = F.pad(h, (0, 0, self.step, 0), mode='circular')
                w = F.pad(w, (self.step, 0, 0, 0), mode='circular')
            else:
                raise NotImplementedError("Invalid pad mode.")
            h = torch.narrow(h, 2, 0, H)
            w = torch.narrow(w, 3, 0, W)
        
        if self.pixel_pad_mode == '0':
            h = F.pad(h, (0, 0, 0, pad_h), "constant", 0)
            w = F.pad(w, (0, pad_w, 0, 0), "constant", 0)
        elif self.pixel_pad_mode == 'c':
            h = F.pad(h, (0, 0, 0, pad_h), mode='circular')
            w = F.pad(w, (0, pad_w, 0, 0), mode='circular')
        elif self.pixel_pad_mode == 'replicate':
            h = F.pad(h, (0, 0, 0, pad_h), mode='replicate')
            w = F.pad(w, (0, pad_w, 0, 0), mode='replicate')
        else:
            raise NotImplementedError("Invalid pad mode.")
        
        h = h.reshape(B, C, (H + pad_h) // self.pixel, self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C*self.pixel, (H + pad_h) // self.pixel, W)
        w = w.reshape(B, C, H, (W + pad_w) // self.pixel, self.pixel).permute(0, 1, 4, 2, 3).reshape(B, C*self.pixel, H, (W + pad_w) // self.pixel)
          
        h = self.mlp_h1(h)
        h = self.mlp_h1_norm(h)
        h = self.act(h)
        h = self.mlp_h2(h)
        
        w = self.mlp_w1(w)
        w = self.mlp_w1_norm(w)
        w = self.act(w)
        w = self.mlp_w2(w)
        
        h = h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C, H + pad_h, W)
        w = w.reshape(B, C, self.pixel, H, (W + pad_w) // self.pixel).permute(0, 1, 3, 4, 2).reshape(B, C, H, W + pad_w)
        
        h = torch.narrow(h, 2, 0, H)
        w = torch.narrow(w, 3, 0, W)
        
        # back to original order
        if self.step and self.step_pad_mode == 'c':
            h = torch.roll(h, -self.step, -2)
            w = torch.roll(w, -self.step, -1)
            
        c = self.mlp_c(x)

        a = (h + w + c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)

        x = h * a[0] + w * a[1] + c * a[2]
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x