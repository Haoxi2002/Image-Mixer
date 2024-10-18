import torch
from matplotlib import pyplot as plt
from torch import nn

from layers.deeplabv3_plus import DeepLab


class Model(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, args=None):
        super().__init__()  # xception mobilenet
        args.modelAda = True
        self.model = DeepLab(num_classes=1, backbone="mobilenet", pretrained=False, downsample_factor=16, image_C=1,
                             dropout=args.dropout, args=args)
        self.EMD = nn.Softmax(dim=-1)
        self.static_embedding = nn.Linear(9, args.pred_len)
        self.flatten = nn.Flatten(start_dim=-2)
        self.channel = args.channel
        if args.channel != 1:
            self.linear1 = nn.Linear(args.channel, 1)
        self.out = nn.Linear(args.seq_len * args.expand * args.h * args.expand, args.pred_len)
        self.linear = nn.Linear(args.pred_len, args.pred_len)

    def work(self, t):
        plt.imshow(t, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.show()

    def forward(self, x, static):
        bs, c, h, w = x.shape
        x = x.view(bs * c, 1, h, w)
        x = self.model(x)
        x = self.EMD(x)
        x = x.view(bs, c, h, w)
        x = self.flatten(x)
        if self.channel != 1:
            x = torch.transpose(x, 1, 2)
            x = self.linear1(x)
            x = torch.transpose(x, 1, 2)
        x = self.out(x)
        x = x * self.static_embedding(static)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x