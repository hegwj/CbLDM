import torch
import torch.nn as nn
from einops import rearrange


class SelfAttention1d(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention1d, self).__init__()
        self.norm = nn.BatchNorm1d(in_channels)
        self.in_channels = in_channels
        self.query = nn.Conv1d(in_channels,
                               in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.key = nn.Conv1d(in_channels,
                               in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.value = nn.Conv1d(in_channels,
                               in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.query(h_)
        k = self.key(h_)
        v = self.value(h_)

        b, c, h = q.shape

        q = rearrange(q, 'b c h -> b (h) c')
        k = rearrange(k, 'b c h -> b c (h)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = rearrange(v, 'b c h-> b c (h)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h) -> b c h', h=h)
        h_ = self.proj_out(h_)

        return x+h_