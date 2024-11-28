import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from collections import OrderedDict
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


class Encoder(nn.Module):
    def __init__(self, input_channel=6, ch=16, latent_channel=2, n_layer=2):
        super().__init__()
        self.input_channel = input_channel
        self.ch = ch
        self.latent_channel = latent_channel
        self.n_layer = n_layer
        self.module = self.net()

    def net(self):
        layers = OrderedDict()

        layers['econv_start'] = nn.Conv1d(in_channels=self.input_channel, out_channels=self.ch,
                                          kernel_size=3, padding=1, stride=1)
        layers['esat_start'] = SelfAttention1d(self.ch)
        layers['eact_sat_start'] = nn.SiLU()
        layers['ebn_start'] = nn.BatchNorm1d(self.ch)

        for i in range(self.n_layer):
            layers[f'econv_d_{i}'] = nn.Conv1d(in_channels=self.ch * 4 ** i, out_channels=self.ch * 4 ** i,
                                               kernel_size=4, padding=1, stride=2)
            layers[f'eact_d_{i}'] = nn.SiLU()
            layers[f'econv_{i}'] = nn.Conv1d(in_channels=self.ch * 4 ** i, out_channels=self.ch * 4 ** i,
                                             kernel_size=3, padding=1, stride=1)
            layers[f'econv_{i}'] = nn.Conv1d(in_channels=self.ch * 4 ** i, out_channels=self.ch * 4 ** (i + 1),
                                             kernel_size=3, padding=1, stride=1)
            layers[f'eact_conv_{i}'] = nn.SiLU()
            layers[f'esat_{i}'] = SelfAttention1d(self.ch * 4 ** (i + 1))
            layers[f'eact_sat_{i}'] = nn.SiLU()
            layers[f'ebn2_{i}'] = nn.BatchNorm1d(self.ch * 4 ** (i + 1))

        layers['rconv_1'] = nn.Conv1d(in_channels=self.ch * 4 ** self.n_layer, out_channels=self.latent_channel,
                                      kernel_size=3, padding=1, stride=1)
        layers['rconv_2'] = nn.Conv1d(in_channels=self.latent_channel, out_channels=self.latent_channel * 2,
                                      kernel_size=3, padding=1, stride=1)

        return nn.Sequential(layers)

    def forward(self, x):
        x = self.module(x)

        return x

    def reparam(self, z):
        mu, log_var = torch.chunk(z, 2, dim=1)
        log_var = nn.functional.softplus(log_var)
        sigma = torch.exp(log_var / 2)

        z_rsample = Independent(Normal(loc=mu, scale=sigma), 2)
        z_sample = z_rsample.rsample()

        return z_sample, mu, log_var


class Decoder(nn.Module):
    def __init__(self, input_channel=6, ch=16, latent_channel=2, n_layer=2):
        super().__init__()
        self.input_channel = input_channel
        self.ch = ch
        self.latent_channel = latent_channel
        self.n_layer = n_layer
        self.module = self.net()

    def net(self):
        layers = OrderedDict()

        layers['dconv_start'] = nn.Conv1d(in_channels=self.latent_channel, out_channels=self.ch * 4 ** self.n_layer,
                                          kernel_size=3, padding=1, stride=1)
        layers['dsat_start'] = SelfAttention1d(self.ch * 4 ** self.n_layer)
        layers['dact_sat_start'] = nn.SiLU()
        layers['dbn_start'] = nn.BatchNorm1d(self.ch * 4 ** self.n_layer)

        for i in range(self.n_layer + 1):
            if i <= self.n_layer - 1:
                layers[f'dt_conv_{i}'] = nn.ConvTranspose1d(in_channels=self.ch * 4 ** (self.n_layer - i),
                                                            out_channels=self.ch * 4 ** (self.n_layer - i),
                                                            kernel_size=2, stride=2, padding=0)
                layers[f'dact_t_{i}'] = nn.SiLU()
                layers[f'dbn_t_{i}'] = nn.BatchNorm1d(self.ch * 4 ** (self.n_layer - i))
                layers[f'dconv_{i}'] = nn.Conv1d(in_channels=self.ch * 4 ** (self.n_layer - i),
                                                 out_channels=self.ch * 4 ** (self.n_layer - i - 1),
                                                 kernel_size=3, padding=1, stride=1)

                layers[f'dact_conv_{i}'] = nn.SiLU()
                layers[f'dsat_{i}'] = SelfAttention1d(self.ch * 4 ** (self.n_layer - i - 1))
                layers[f'dact_sat_{i}'] = nn.SiLU()
                layers[f'dbn_{i}'] = nn.BatchNorm1d(self.ch * 4 ** (self.n_layer - i - 1))

            else:
                layers[f'dconv_{i}'] = nn.Conv1d(in_channels=self.ch,
                                                 out_channels=self.ch,
                                                 kernel_size=3, padding=1, stride=1)
                layers[f'dact_conv_{i}'] = nn.SiLU()

        layers['dconv_-1'] = nn.Conv1d(in_channels=self.ch,
                                       out_channels=self.input_channel,
                                       kernel_size=3, padding=1, stride=1)
        layers['dconv_-2'] = nn.Conv1d(in_channels=self.input_channel,
                                       out_channels=self.input_channel,
                                       kernel_size=3, padding=1, stride=1)
        return nn.Sequential(layers)

    def forward(self, x):
        x = self.module(x)

        return x
