import torch
from torch import nn


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, token_mlp_dim, channel_mlp_dim):
        super(MixerBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.mlp_token = MlpBlock(token_dim, token_mlp_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp_channel = MlpBlock(hidden_dim, channel_mlp_dim)

    def forward(self, x):
        y = self.layer_norm_1(x)  # (bs, patches, c)
        y = torch.transpose(y, -1, -2)  # (bs, c, patches)
        y = self.mlp_token(y)
        y = torch.transpose(y, -1, -2)
        x = x + y  # (bs, patches, c)
        y = self.layer_norm_2(x)
        y = self.mlp_channel(y)
        x = x + y

        return x
