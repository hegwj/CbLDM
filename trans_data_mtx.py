import h5py
import torch
from tqdm import tqdm
import os
import numpy as np


def trans_data(root):
    h = h5py.File(root)

    xyz = h["Node Feature Matrix"][:]
    pdf_cut = h["PDF label"][:]

    h.close()

    length = len(pdf_cut)
    pdf = np.zeros(shape=[3000])

    pdf[:length] = pdf_cut
    pdf = pdf / np.max(pdf)

    num = xyz.shape[0]
    xyz = torch.from_numpy(xyz)

    NFM_o = torch.exp(-1 * torch.norm(xyz[:, None] - xyz, dim=2, p=2) ** 2 / 260)
    NFM = torch.zeros(size=[256, 256])
    NFM[:num, :num] = NFM_o

    NFM_4 = torch.zeros(size=[4, 128, 128])
    NFM_4[0, :, :] = NFM[0:128, 0:128]
    NFM_4[1, :, :] = NFM[0:128, 128:256]
    NFM_4[2, :, :] = NFM[128:256, 0:128]
    NFM_4[3, :, :] = NFM[128:256, 128:256]

    pdf6 = np.zeros(shape=[6, 500])
    for i in range(6):
        if i != 5:
            pdf6[i, :] = pdf[i * 500: (i + 1) * 500]
        else:
            pdf6[5, :] = pdf[i * 500:]

    return NFM_4, pdf6


if __name__ == "__main__":
    path = 'your_path'
    path_save = 'your_save_path'

    for file in tqdm(os.listdir(path)):
        mtx, pdf = trans_data(path + file)
        h = h5py.File(path_save + file, 'w')
        h.create_dataset('Matrix', data=mtx.cpu().detach().numpy())
        h.create_dataset('PDF', data=pdf)
        h.close()
