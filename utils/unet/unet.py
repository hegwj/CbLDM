import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.actfun import swish
from utils.attn import SpatialTransformer as SplAttn


def get_timestep_embedding(timestep, dim):
    emb = math.log(10000) / (dim // 2 - 1)
    emb = torch.exp(torch.arange(dim // 2, dtype=torch.float32) * -emb).to(device=timestep.device)
    temb = timestep.float()[:, None] * emb[None, :]
    temb = torch.cat([torch.sin(temb), torch.cos(temb)], dim=1)
    if dim % 2 == 1:
        temb = torch.nn.functional.pad(temb, (0,1,0,0))
    return temb


def Normalize(in_channels, num_groups=4):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels, input_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, input_channels, output_channels=None, temb_channels=512):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels if output_channels is not None else input_channels

        self.swish = swish()
        self.norm1 = Normalize(self.input_channels)
        self.conv1 = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, self.output_channels) if temb_channels > 0 else None
        self.norm2 = Normalize(self.output_channels)
        self.conv2 = nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        if self.input_channels != self.output_channels:
            self.nin_shortcut = torch.nn.Conv2d(self.input_channels, self.output_channels,
                                                kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.swish(h)
        h = self.conv2(h)

        if self.input_channels != self.output_channels:
            x = self.nin_shortcut(x)

        return x+h


class UNet(nn.Module):
    def __init__(self,input_channels, output_channels, context_channels, ch, ch_mult=(1,2), num_res_blocks=2):
        super(UNet, self).__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.input_channels = input_channels
        self.context_channels = context_channels

        self.down_block_chans = []
        self.swish = swish()
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([torch.nn.Linear(self.ch, self.temb_ch),
                                         torch.nn.Linear(self.temb_ch, self.temb_ch)])

        in_ch_mult = (1,) + tuple(ch_mult)

        # downsampling
        self.conv_in = torch.nn.Conv2d(self.input_channels, self.ch, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(input_channels=block_in, output_channels=block_out, temb_channels=self.temb_ch))
                block_in = block_out
                attn.append(SplAttn(input_channels=block_out, context_dim=self.context_channels))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(input_channels=block_in, temb_channels=self.temb_ch)
        self.mid.attn_1 = SplAttn(input_channels=block_in,context_dim=self.context_channels)
        self.mid.block_2 = ResnetBlock(input_channels=block_in, temb_channels=self.temb_ch)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(input_channels=block_in + skip_in, output_channels=block_out,
                                         temb_channels=self.temb_ch))
                block_in = block_out
                attn.append(SplAttn(input_channels=block_out, context_dim=self.context_channels))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, context, t):
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = self.swish(temb)
        temb = self.temb.dense[1](temb)

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h, context)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h,context)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h,context)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h