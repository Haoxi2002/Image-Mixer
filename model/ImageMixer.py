import einops
import torch
from torch import nn

from layers.mixer import MixerBlock, MlpBlock

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.token_dim = (args.h * args.expand // args.patch_size[0]) * (args.seq_len * args.expand // args.patch_size[1])  # token <==> patch
        self.conv_embedding = nn.Conv2d(args.channel, args.hidden_dim, stride=args.patch_size, kernel_size=args.patch_size, padding=0)
        self.embedding_feedforward = MlpBlock(args.hidden_dim, args.hidden_dim * 2, args.dropout)
        self.static_embedding1 = nn.Linear(9, self.token_dim * args.hidden_dim)
        self.fc_fusion1 = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.blocks = nn.ModuleList(
            [MixerBlock(args.hidden_dim, self.token_dim, args.token_mlp_dim, args.channel_mlp_dim, args.dropout) for _
             in range(args.n_blocks)])
        self.head_layer_norm = nn.LayerNorm(args.hidden_dim)
        self.flatten = nn.Flatten(start_dim=-2)
        self.static_embedding2 = nn.Linear(9, self.token_dim * args.hidden_dim)
        self.fc_fusion2 = nn.Linear(self.token_dim * args.hidden_dim * 2, self.token_dim * args.hidden_dim)
        self.linear = nn.Linear(self.token_dim * args.hidden_dim, args.pred_len)

    """
    input:    
        x: (batch_size, channel*features, h, seq_len)
        static: (batch_size, features, 9)
    output:
        seq_y: (batch_size, pred_len, features)   
    """

    def forward(self, x, static):
        bc, f_c, h, w = x.shape
        # CI
        x = torch.reshape(x, (-1, self.args.channel, h, w))
        static = torch.reshape(static, (-1, 1, 9))

        # encoder
        x = self.conv_embedding(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        # early fusion
        # x = self.embedding_feedforward(x)
        # e_static = self.static_embedding1(static)
        # e_static = torch.reshape(e_static, (bc, self.token_dim, self.args.hidden_dim))
        # x = self.fc_fusion1(torch.cat((x, e_static), dim=2))

        # backbone
        for l in self.blocks:
            x = l(x)

        # decoder (b, p, c)  b=batch_size p=patches c=channel
        # x = self.head_layer_norm(x)
        x = self.flatten(x)
        x = torch.unsqueeze(x, dim=1)
        # Late fusion
        # d_static = self.static_embedding2(static)
        # x = torch.cat([d_static, x], dim=2)
        # x = self.fc_fusion2(x)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)  # (bs, pred_len, 1)

        # de CI
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (bc, f_c // self.args.channel, self.args.pred_len))
        x = torch.transpose(x, 1, 2)

        return x.float()
