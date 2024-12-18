import os, torch, h5py, random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl


class Fulldl(pl.LightningDataModule):
    def __init__(self, data_dir='your_data_path',
                 num_files=None, batch_size=32, shuffle=True, num_workers=0):
        super().__init__()
        self.batch_size = int(batch_size)
        self.num_workers = num_workers
        self.files_sorted = sorted(os.listdir(data_dir))

        files = self.files_sorted.copy()

        if shuffle:
            random.shuffle(files)

        if files is not None:
            files = files[:num_files]
        else:
            pass

        # n_train = int(0.8 * len(files))
        n_train = int(0.95 * len(files))
        n_valid = int((len(files) - n_train))

        print('\nBatch size: {}'.format(batch_size))
        print('Total number of graphs {}.'.format(len(files)))
        print('\tTraining files:', n_train)
        print('\tValidation files:', n_valid)

        self.trSamples, self.vlSamples = list(), list()
        print('Loading graphs:')

        for idx in tqdm(range(len(files))):
            h5f = h5py.File(data_dir + '/' + files[idx], 'r')
            a = h5f['Matrix'][:]
            b = h5f['PDF'][:]
            h5f.close()

            d = np.zeros([3000])
            for i in range(6):
                d[500 * i: 500 * (i + 1)] = b[i, :]
            d = torch.tensor(d, dtype=torch.float)

            m = torch.tensor(a, dtype=torch.float)
            # p = torch.tensor(b / np.max(np.abs(b)), dtype=torch.float)
            p = torch.tensor(b, dtype=torch.float)
            name_idx = torch.tensor(self.files_sorted.index(files[idx]), dtype=torch.int16)

            if idx < n_train:
                self.trSamples.append(
                    tuple((m, p, d, name_idx)))
            elif idx < n_train + n_valid:
                self.vlSamples.append(
                    tuple((m, p, d, name_idx)))

    def train_dataloader(self):
        return DataLoader(self.trSamples, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.vlSamples, batch_size=self.batch_size, num_workers=self.num_workers)
