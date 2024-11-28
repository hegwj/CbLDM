import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim=3000, latent_dim=256, nhead=32):
        super(MultiHeadAttention, self).__init__()
        assert latent_dim % nhead == 0

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.d_k = latent_dim // nhead
        self.nhead = nhead

        self.bn = nn.BatchNorm1d(input_dim)
        self.query = nn.Linear(input_dim, latent_dim)
        self.key = nn.Linear(input_dim, latent_dim)
        self.value = nn.Linear(input_dim, latent_dim)

        self.out = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        bs = x.shape[0]

        h_ = self.bn(x)

        q = self.query(h_).view(bs, self.nhead, self.d_k).transpose(1, 2)
        k = self.key(h_).view(bs, self.nhead, self.d_k).transpose(1, 2)
        v = self.value(h_).view(bs, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** .5
        attn = torch.softmax(scores, dim=1)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, self.latent_dim)

        output = self.out(output)

        return output
