import numpy as np
from einops import rearrange
import torch.nn as nn
from layers.deeplabv3_plus import DeepLab
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, args):
        super().__init__()  #xception mobilenet
        args.modelAda = True
        self.model = DeepLab(num_classes=1, backbone="mobilenet", pretrained=False, downsample_factor=16, image_C=1,
                             dropout=args.dropout, args=args)
        self.conv = nn.Conv2d(in_channels=args.channel, out_channels=1, kernel_size=args.expand, padding=0, bias=False, stride=args.expand)
        self.EMD = nn.Softmax(dim=-1)
        self.expand = args.expand
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(args.seq_len * args.expand * args.seq_len * 2 * args.expand, args.pred_len)

    def forward(self, x, static):
        bs, c, w, h = x.shape
        x = x.view(bs * c, 1, w, h)
        x = self.model(x)
        x = x.view(bs, c, w, h)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = torch.transpose(x, -1, -2)
        return x