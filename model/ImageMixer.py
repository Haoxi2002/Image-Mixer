import einops
import torch
from torch import nn

from layers.common import MixerBlock, MlpBlock


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.token_dim = (args.seq_len * args.expand // args.patch_size[0]) * (args.h * args.expand // args.patch_size[1]) + 1  # token <==> patch
        self.conv_embedding = nn.Conv2d(args.channel, args.hidden_dim, stride=args.patch_size, kernel_size=args.patch_size, padding=0)
        self.static_embedding = nn.Linear(4, args.hidden_dim)
        self.blocks = nn.ModuleList([MixerBlock(args.hidden_dim, self.token_dim, args.token_mlp_dim, args.channel_mlp_dim, args.dropout) for _ in range(args.n_blocks)])
        self.head_layer_norm = nn.LayerNorm(args.hidden_dim)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.token_dim * args.hidden_dim, args.pred_len)

    """
    input:    
        fig_x: (batch_size, channel*features, seq_len, h)
    output:
        seq_y: (batch_size, pred_len, features)   
    """

    def forward(self, x, static):

        bc, f_c, l, h = x.shape
        # CI
        x = torch.reshape(x, (-1, self.args.channel, x.shape[2], x.shape[3]))
        static = torch.reshape(static, (-1, 1, 4))

        # encoder
        x = self.conv_embedding(x.float())
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        static = self.static_embedding(static.float())
        x = torch.cat([static, x], 1)

        # backbone
        for l in self.blocks:
            x = l(x)

        # decoder (b, p, c)  b=batch_size p=patches c=channel
        x = self.flatten(x)
        x = self.linear(x)
        x = torch.unsqueeze(x, dim=-1)

        # de CI
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (bc, f_c // self.args.channel, self.args.pred_len))
        x = torch.transpose(x, 1, 2)

        return x.float()
