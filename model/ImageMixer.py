import einops
import torch
from torch import nn

from layers.common import MixerBlock, MlpBlock


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.token_dim = (args.seq_len * args.expand // args.patch_size[0]) * (args.h * args.expand // args.patch_size[1])  # token <==> patch
        self.conv_embedding = nn.Conv2d(args.channel, args.hidden_dim, stride=args.patch_size, kernel_size=args.patch_size, padding=0)
        self.blocks = nn.ModuleList([MixerBlock(args.hidden_dim, self.token_dim, args.token_mlp_dim, args.channel_mlp_dim) for _ in range(args.n_blocks)])
        self.head_layer_norm = nn.LayerNorm(args.hidden_dim)
        self.linear1 = nn.Linear(args.hidden_dim, 1)
        self.linear2 = nn.Linear(self.token_dim, args.pred_len)
        self.linear3 = MlpBlock(args.pred_len, args.pred_len * 2)

    """
    input:    
        fig_x: (batch_size, channel*features, seq_len, h)
    output:
        seq_y: (batch_size, pred_len, features)   
    """

    def forward(self, x, mean):

        bc, f_c, l, h = x.shape
        # CI
        x = torch.reshape(x, (-1, self.args.channel, x.shape[2], x.shape[3]))

        # encoder
        x = self.conv_embedding(x.float())
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        # backbone
        for l in self.blocks:
            x = l(x)

        # decoder
        x = self.head_layer_norm(x)  # (b, p, c)  b=batch_size p=patches c=channel
        x = self.linear1(x)
        x = einops.rearrange(x, 'b p f -> b f p')  # b=batch_size f=features p=patches
        x = self.linear2(x)
        x = einops.rearrange(x, 'b f l -> b l f')  # b=batch_size f=features l=pred_len

        # de CI
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (bc, f_c // self.args.channel, self.args.pred_len))
        x = torch.transpose(x, 1, 2)

        x = einops.rearrange(x, 'b l f -> b f l')
        x = self.linear3(x)
        x = einops.rearrange(x, 'b f l -> b l f')

        x = x * mean
        return x.float()
