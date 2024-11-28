import torch
import torch.nn as nn


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)