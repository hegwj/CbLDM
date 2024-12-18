import random
import time
import h5py
import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.distributions.kl import kl_divergence as KLD
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.optim import Adan
from modules.vae4io_pdf import Encoder, Decoder
from modules.dataloader_io import Mtx

seed = 42
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.manual_seed(seed)
random.seed(seed)


class Net(pl.LightningModule):
    def __init__(self, *, lr=5e-4, wd=1e-5, acc=True):
        super(Net, self).__init__()
        self.lr = lr
        self.wd = wd
        self.threshold = 0.8
        self.threshold_high = 2
        self.beta = 0.
        self.update = 0
        self.count = 0
        self.cold_down = 0

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.reparam = Encoder().reparam

        self.m = torch.tensor(Matrix, dtype=torch.float32, device='cuda')
        self.pdf = torch.tensor(p, dtype=torch.float32, device='cuda')

    def forward(self, z, pdf):
        z, prior, pdf_prior = self.encoder(z, pdf)
        _, pdf_log_var = torch.chunk(pdf_prior, 2, dim=1)
        pdf_log_var = torch.nn.functional.softplus(pdf_log_var)
        pdf_sigma = torch.exp(pdf_log_var / 2)
        z_sample, posterior = self.reparam(z)

        z_pre = self.decoder(z_sample)

        return z_pre, prior, posterior, pdf_sigma

    def configure_optimizers(self):
        step = list(self.encoder.parameters()) + list(self.decoder.parameters())
        return Adan(step, lr=self.lr, weight_decay=self.wd)

    def training_step(self, batch, batch_nb):
        z = batch[0].clone()
        pdf = batch[1].clone()
        z_pre, prior, posterior, pdf_sigma = self.forward(z, pdf)

        kl = torch.sum(KLD(posterior, prior)) / z.shape[0]
        mse = F.mse_loss(z * 1000, z_pre * 1000)
        loss = mse + self.beta * 0.1 * kl

        self.log('train_mdf_mse', mse, prog_bar=True, on_step=True, on_epoch=False, enable_graph=True)
        self.log('train_kl', kl, prog_bar=True, on_step=True, on_epoch=False, enable_graph=True)

        return loss

    def validation_step(self, batch, batch_nb):
        z = batch[0].clone()
        pdf = batch[1].clone()
        z_pre, prior, posterior, _ = self.forward(z, pdf)

        kl = torch.sum(KLD(posterior, prior)) / z.shape[0]
        loss = F.mse_loss(z * 1000, z_pre * 1000)
        mse = F.mse_loss(z, z_pre)

        if self.update != self.current_epoch and mse < self.threshold and self.beta < 1:
            if self.beta < 0.05:
                self.beta += 0.0001
            elif 0.05 <= self.beta < 0.3:
                self.beta += 0.0003
            elif 0.3 <= self.beta < 0.55:
                self.beta += 0.001
            else:
                self.beta += 0.005

            self.update = self.current_epoch

        if mse > self.threshold_high:
            self.count += 1
            if self.count >= 10:
                self.beta = self.beta * 0.999
                self.count = 0

        self.log('vld_mdf_mse', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('vld_mse', mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('vld_kl', kl, prog_bar=True, on_step=False, on_epoch=True)
        self.log('beta', self.beta, prog_bar=True, on_step=False, on_epoch=True)

        return loss


if __name__ == '__main__':
    start_time = time.time()

    num = 0

    if os.path.exists(f'./models/{num}') != 1:
        os.mkdir(f'./models/{num}')

    if os.path.exists(f'./models/{num}/pred') != 1:
        os.mkdir(f'./models/{num}/pred')

    acc_mode = True
    mtx = Mtx(data_dir='your_data_path', batch_size=64)
    model = Net(lr=2e-4, wd=2e-4, acc=acc_mode)
    print(model)

    callback0 = ModelCheckpoint(
        monitor='vld_mdf_mse',
        dirpath=f'./models/{num}',
        filename='model_{vld_mdf_mse:.3f}',
        save_top_k=1,
        mode='min',
        save_last=False,
    )
    callback1 = ModelCheckpoint(
        monitor='beta',
        dirpath=f'./models/{num}',
        filename='model_{beta:.5f}',
        save_top_k=1,
        mode='max',
        save_last=False,
    )
    callback2 = ModelCheckpoint(
        monitor='acc',
        dirpath=f'./models/{num}',
        filename='model_{acc:.5f}',
        save_top_k=1,
        mode='min',
        save_last=False,
    )

    callback3 = ModelCheckpoint(
        dirpath=f'./models/{num}',
        filename='model_epoch_{epoch:05d}',
        save_top_k=1,
        save_last=False,
        every_n_epochs=50,
    )

    if acc_mode:
        callback = [callback0, callback1, callback2, callback3]
    else:
        callback = [callback0, callback1, callback3]

    csv_l = CSVLogger(f'./models/{num}')
    tb_l = TensorBoardLogger(f'./models/{num}')

    trainer = pl.Trainer(accelerator='gpu', precision='16-mixed',
                         max_epochs=10240, callbacks=callback, logger=tb_l,
                         log_every_n_steps=10, reload_dataloaders_every_n_epochs=100)
    trainer.fit(model, mtx)
    trainer.save_checkpoint(f'./models/{num}/last.ckpt')

    end_time = time.time()
    print(f'took: {(end_time - start_time) / 60:.2f} min')
