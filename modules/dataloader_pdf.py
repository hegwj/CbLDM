import os, torch, h5py, random
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader


class PDF(pl.LightningDataModule):
    def __init__(self, data_dir='E:/useful/Project/Pro_Diff/test2/data',
                 num_files=None, batch_size=64, shuffle=True, num_workers=0):
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

        n_train = int(0.65 * len(files))
        n_valid = int((len(files) - n_train))

        print('\nBatch size: {}'.format(batch_size))
        print('Total number of graphs {}.'.format(len(files)))
        print('\tTraining files:', n_train)
        print('\tValidation files:', n_valid)

        self.trSamples, self.vlSamples = list(), list()
        print('Loading graphs:')

        for idx in tqdm(range(len(files))):
            h5f = h5py.File(data_dir + '/' + files[idx], 'r')
            a = h5f['PDF_x'][:]
            b = h5f['PDF_y'][:]
            h5f.close()

            m = torch.tensor(a / np.max(np.abs(a)), dtype=torch.float)
            n = torch.tensor(b / np.max(np.abs(b)), dtype=torch.float)
            name_idx = torch.tensor(self.files_sorted.index(files[idx]), dtype=torch.int32)

            if idx < n_train:
                self.trSamples.append(
                    tuple((m, n, name_idx)))
            elif idx < n_train + n_valid:
                self.vlSamples.append(
                    tuple((m, n, name_idx)))

    def train_dataloader(self):
        return DataLoader(self.trSamples, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.vlSamples, batch_size=self.batch_size, num_workers=self.num_workers)
