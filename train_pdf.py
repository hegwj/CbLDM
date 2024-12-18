import random, time, os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.optim import Adan
from modules.vae4con import Encoder, Decoder
from modules.dataloader_pdf import PDF


seed = 42
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.manual_seed(seed)
random.seed(seed)

def KLD(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


class Net(pl.LightningModule):
    def __init__(self, *, lr=5e-4, wd=1e-5, alpha=1e-1):
        super(Net, self).__init__()
        self.lr = lr
        self.wd = wd
        self.threshold = 0.003
        self.threshold_high = 0.006
        self.alpha = alpha
        self.beta = 0
        self.update = 0
        self.update2 = 0

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.reparam = Encoder().reparam

    def forward(self, data):
        z = data
        z = self.encoder(z)
        z_sample, mu, log_var = self.reparam(z)
        z_pre = self.decoder(z_sample)

        return z_pre, mu, log_var

    def configure_optimizers(self):
        step = list(self.encoder.parameters()) + list(self.decoder.parameters())
        return Adan(step, lr=self.lr, weight_decay=self.wd)

    def training_step(self, batch, batch_nb):
        z = batch[0].clone()
        t = batch[0].clone()
        z_pre, mu, log_var = self.forward(z)

        kl = KLD(mu, log_var) / z.size(0)
        loss = (F.l1_loss(t * 100, z_pre * 100) + 0.1 * F.mse_loss(t * 100, z_pre * 100) +
                self.beta * self.alpha * kl)
        l1 = F.l1_loss(t, z_pre)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, enable_graph=True)
        self.log('train_l1', l1, prog_bar=True, on_step=True, on_epoch=False, enable_graph=True)
        self.log('train_kl', kl, prog_bar=True, on_step=True, on_epoch=False, enable_graph=True)
        return loss

    def validation_step(self, batch, batch_nb):
        z = batch[0].clone()
        t = batch[0].clone()
        z_pre, mu, log_var = self.forward(z)

        kl = KLD(mu, log_var) / z.size(0)
        loss = (F.l1_loss(t * 100, z_pre * 100) + 0.1 * F.mse_loss(t * 100, z_pre * 100) +
                self.beta * self.alpha * kl)
        l1 = F.l1_loss(t, z_pre)
        mdf_l1 = F.l1_loss(t * 100, z_pre * 100)

        if self.update != self.current_epoch and l1 < self.threshold:
            if self.beta < 0.05:
                self.beta += 0.001
            elif 0.05 <= self.beta < 0.3:
                self.beta += 0.003
            elif 0.3 <= self.beta < 0.55:
                self.beta += 0.01
            else:
                self.beta += 0.02

            self.update = self.current_epoch

        if self.update2 != self.current_epoch and l1 > self.threshold_high and self.beta > 0.3:
            self.beta = self.beta * 0.999
            self.update2 = self.current_epoch

        self.log('vld_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('vld_l1', l1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('vld_kl', kl, prog_bar=True, on_step=False, on_epoch=True)
        self.log('beta', self.beta, prog_bar=True, on_step=False, on_epoch=True)

        return loss


if __name__ == '__main__':
    model_num = 0

    if os.path.exists(f'./models/{model_num}') != 1:
        os.mkdir(f'./models/{model_num}')

    if os.path.exists(f'./models/{model_num}/pred') != 1:
        os.mkdir(f'./models/{model_num}/pred')

    start_time = time.time()

    pdf = PDF(data_dir='your_pdf_path', batch_size=128)
    model = Net(lr=5e-5, wd=1e-5, alpha=1e-3)
    print(model)

    callback0 = ModelCheckpoint(
        monitor='vld_loss',
        dirpath=f'./models/{model_num}',
        filename='model_{vld_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=False,
    )
    callback1 = ModelCheckpoint(
        monitor='beta',
        dirpath=f'./models/{model_num}',
        filename='model_{beta:.5f}',
        save_top_k=1,
        mode='max',
        save_last=False,
    )

    callback2 = ModelCheckpoint(
        monitor='vld_l1',
        dirpath=f'./models/{model_num}',
        filename='model_{vld_l1:.7f}',
        save_top_k=1,
        mode='min',
        save_last=False,
    )

    callback3 = ModelCheckpoint(
        monitor='vld_kl',
        dirpath=f'./models/{model_num}',
        filename='model_{vld_kl:.2f}',
        save_top_k=1,
        mode='min',
        save_last=False,
    )

    callback4 = ModelCheckpoint(
        dirpath=f'./models/{model_num}',
        filename='model_epoch_{epoch:05d}',
        save_top_k=1,
        save_last=False,
        every_n_epochs=500,
    )

    callback = [callback0, callback1, callback2, callback4]

    csv_l = CSVLogger(f'./models/{model_num}')
    tb_l = TensorBoardLogger(f'./models/{model_num}')

    trainer = pl.Trainer(accelerator='gpu', precision='32-true',
                         max_epochs=131072, callbacks=callback, logger=tb_l, reload_dataloaders_every_n_epochs=300)
    trainer.fit(model, pdf)
    trainer.save_checkpoint(f'./models/{model_num}/last.ckpt')

    end_time = time.time()
    print(f'took: {(end_time - start_time) / 60:.2f} min')
