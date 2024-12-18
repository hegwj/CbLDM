import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Normal, Independent

from utils.optim import Adan

from modules.vae4io_pdf import Encoder as io_Encoder
from modules.vae4io_pdf import Decoder as io_Decoder
from modules.vae4con import Encoder as pdf_Encoder
from modules.vae4con import Decoder as pdf_Decoder

from utils.unet import UNetModel
from modules.ldm import GaussianDiffusion

class Net_io(pl.LightningModule):
    def __init__(self, z_ch=1):
        super(Net_io, self).__init__()
        self.encoder = io_Encoder(z_channels=z_ch)
        self.decoder = io_Decoder(z_channels=z_ch)
        self.reparam = io_Encoder(z_channels=z_ch).reparam

    def forward(self, data):
        z = data
        z = self.encoder(z)
        z_sample, mu, log_var = self.reparam(z)
        z_pre = self.decoder(z_sample)

        return z_pre, mu, log_var

class Net_pdf(pl.LightningModule):
    def __init__(self):
        super(Net_pdf, self).__init__()
        self.encoder = pdf_Encoder()
        self.decoder = pdf_Decoder()
        self.reparam = pdf_Encoder().reparam

    def forward(self, data):
        z = data
        z = self.encoder(z)
        z_sample, mu, log_var = self.reparam(z)
        z_pre = self.decoder(z_sample)

        return z_pre, mu, log_var


class Net_ldm(pl.LightningModule):
    def __init__(self, *, input_channels=1, context_dim=2 * 125, ch=128, ch_mult=(1, 2, 2, 4), attn_res=(2, 4, 8),
                 num_res_blocks=2, transformer_depth=2,
                 timestep=1000, timescale=1000,
                 x_scale=1., beta_schedule='cosine',
                 lr=2e-4, wd=2e-4, bs=32,
                 path_io=None, path_pdf=None):

        super(Net_ldm, self).__init__()
        self.unet = UNetModel(in_channels=input_channels, context_dim=context_dim,
                              model_channels=ch, channel_mult=ch_mult, attention_resolutions=attn_res,
                              num_res_blocks=num_res_blocks, transformer_depth=transformer_depth)
        self.g_noise = GaussianDiffusion(timestep=timestep, timescale=timescale,
                                         beta_schedule=beta_schedule)

        self.lr = lr
        self.wd = wd
        self.batch_size = bs
        self.x_scale = x_scale
        self.timestep = timestep
        self.in_ch = input_channels

        self.path_io = path_io
        self.path_pdf = path_pdf

        self.model_vae_en = Net_io.load_from_checkpoint(self.path_io).eval().encoder
        self.model_vae_de = Net_io.load_from_checkpoint(self.path_io).eval().decoder
        self.model_pdf_en = Net_pdf.load_from_checkpoint(self.path_pdf).eval().encoder

        for param in self.model_vae_en.parameters():
            param.requires_grad = False
        for param in self.model_vae_de.parameters():
            param.requires_grad = False
        for param in self.model_pdf_en.parameters():
            param.requires_grad = False

    @staticmethod
    def reparam(z):
        mu, log_var = torch.chunk(z, 2, dim=1)
        log_var = nn.functional.softplus(log_var)

        sigma = torch.exp(log_var / 2)
        z_rsample = Independent(Normal(loc=mu, scale=sigma), 2)
        z_sample = z_rsample.rsample()

        return z_sample

    def forward(self, data, noise=None, pred=False, focus=False, reparam=False, ts=1000, prt=None):
        if not pred:
            pdf_io = data[2].clone()
            pdf = data[1].clone()
            mat = data[0].clone()

            x, _, _ = self.model_vae_en(mat, pdf_io)

            if reparam:
                x = self.reparam(x)
            else:
                x = x[:, 0: x.shape[1] // 2, :, :]

            x /= self.x_scale

            con = self.model_pdf_en(pdf)

            con = con[:, 0: con.shape[1] // 2, :]

            con = rearrange(con, 'b c h -> b 1 (c h)')

            t = torch.randint(0, self.timestep - 1, (x.shape[0],), device=x.device)
            noise = torch.randn_like(x) if noise is None else noise

            x_t = self.g_noise.q_sample(x, t, noise)
            pred_noise = self.unet(x_t, con, t)

            return noise, pred_noise

        else:
            pdf = data.clone()
            if prt is not None:
                mat = torch.randn((pdf.shape[0], 4, 128, 128)).to(pdf.device)
                _, _, x = self.model_vae_en(mat, pdf, pred=True)

            # x = self.model_vae_en(mat)
            # x = x[:, 0: x.shape[1] // 2, :, :]
            # x /= self.x_scale

            con = self.model_pdf_en(pdf)

            con = con[:, 0: con.shape[1] // 2, :]

            con = rearrange(con, 'b c h -> b 1 (c h)')

            # t = torch.full((x.shape[0],), fill_value=self.timestep - 1, device=x.device)
            # x_t = self.g_noise.q_sample(x, t=t)
            if prt is not None:
                x_pred, x_preds = self.g_noise.p_sample_loop(self.unet, x, con, self.x_scale, focus,
                                                             timestep=ts, time_skip=self.timestep - prt)

            else:
                x_t = torch.randn((pdf.shape[0], self.in_ch, 16, 16)).to(pdf.device)
                # x_pred, x_preds = self.g_noise.p_sample_loop(self.unet, x_t, con, self.x_scale, focus)
                x_pred, x_preds = self.g_noise.p_sample_loop(self.unet, x_t, con,
                                                             x_scale=self.x_scale, focus=focus)

            if focus:
                x_preds_hat = []
                for i in range(len(x_preds)):
                    x_hat = self.model_vae_de(x_preds[i])
                    x_preds_hat.append(x_hat)
            else:
                x_pred = self.model_vae_de(x_pred)

            return x_pred if not focus else x_preds_hat

    def configure_optimizers(self):
        return torch.optim.AdamW(self.unet.parameters(), lr=self.lr, weight_decay=self.wd)

    def training_step(self, batch, batch_nb):
        noise, pred_noise = self.forward(batch, reparam=True)

        # loss = F.smooth_l1_loss(pred_noise, noise, beta=0.15)
        mse = F.mse_loss(pred_noise, noise)
        l1 = F.l1_loss(pred_noise, noise)
        loss = l1 if mse > l1 / 2 else mse

        self.log('tr_loss', loss,
                 prog_bar=True, on_step=True, on_epoch=False, batch_size=self.batch_size)
        self.log('tr_mse', mse,
                 prog_bar=True, on_step=True, on_epoch=False, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_nb):
        noise, pred_noise = self.forward(batch, reparam=False)

        # loss = F.smooth_l1_loss(pred_noise, noise, beta=0.15)
        mse = F.mse_loss(pred_noise, noise)
        l1 = F.l1_loss(pred_noise, noise)
        loss = l1 if mse > l1 / 2 else mse

        self.log('val_loss', loss,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val_mse', mse,
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return loss
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        model = cls()
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        return model

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_vae_en_state_dict'] = None
        checkpoint['model_vae_de_state_dict'] = None
        checkpoint['model_pdf_en_state_dict'] = None
        return checkpoint
