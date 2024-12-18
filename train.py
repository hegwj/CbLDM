import torch
import random
import os
import time
import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.vae_ldm import Net_ldm as Net
from modules.dataloader import Fulldl

seed = 42
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.manual_seed(seed)
random.seed(seed)

if __name__ == '__main__':
    model_num = 0
    model_path = 'your_model_path'
    predict_path = 'your_predict_save_path'
    if os.path.exists(model_path) != 1:
        os.mkdir(model_path)

    if os.path.exists(predict_path) != 1:
        os.mkdir(predict_path)

    start_time = time.time()

    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 0.
    beta = 'cosine'  # modified
    data_path = 'your_data_path'
    data = Fulldl(data_dir=data_path, batch_size=batch_size)
    model = Net(bs=batch_size, lr=learning_rate, wd=weight_decay, beta_schedule=beta)
    print(model)

    csv_l = CSVLogger(f'./models/{model_num}')
    tb_l = TensorBoardLogger(f'./models/{model_num}')

    callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'./models/{model_num}',
        filename='model_{val_loss:.5f}',
        save_top_k=1,
        mode='min',
        save_last=False,
    )

    callback1 = ModelCheckpoint(
        dirpath=f'./models/{model_num}',
        filename='model_epoch_{epoch:05d}',
        save_top_k=1,
        save_last=False,
        every_n_epochs=50,
    )

    trainer = pl.Trainer(accelerator='gpu', precision='32-true',
                         max_epochs=2048, callbacks=[callback, callback1],
                         logger=tb_l, log_every_n_steps=5, reload_dataloaders_every_n_epochs=50)
    trainer.fit(model, data)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    trainer.save_checkpoint(f'./models/{model_num}/last.ckpt')

    end_time = time.time()

    print('\n')
    print(f'took: {(end_time - start_time) / 60:.2f} min')
