import h5py
import torch
import os
import numpy as np
from tqdm import tqdm


def trans_data(root):
    h = h5py.File(root)

    xyz = h["Node Feature Matrix"][:]
    num = xyz.shape[0]
    xyz = torch.from_numpy(xyz)

    NFM_o = torch.norm(xyz[:, None] - xyz, dim=2, p=2)
    NFM = torch.zeros(size=[256, 256]) - 10

    NFM[:num, :num] = NFM_o

    NFM_4 = torch.zeros(size=[4, 128, 128])
    NFM_4[0, :, :] = NFM[0:128, 0:128]
    NFM_4[1, :, :] = NFM[0:128, 128:256]
    NFM_4[2, :, :] = NFM[128:256, 0:128]
    NFM_4[3, :, :] = NFM[128:256, 128:256]

    return NFM_4

def trans_data2(root):
    h = h5py.File(root)

    pdf_cut = h["PDF label"][:]
    length = len(pdf_cut)
    pdf = np.zeros(shape=[3000])

    pdf[:length] = pdf_cut
    pdf = pdf / np.max(pdf)

    pdf6 = np.zeros(shape=[6, 500])
    for i in range(6):
        if i != 5:
            pdf6[i, :] = pdf[i * 500: (i + 1) * 500]
        else:
            pdf6[5, :] = pdf[i * 500:]

    return pdf6


if __name__ == "__main__":
    path = 'original_data_path'
    path_save = 'save_data_path'

    for file in tqdm(os.listdir(path)):
        mtx = trans_data(path + file)
        pdf = trans_data2(path + file)
        h = h5py.File(path_save + file, 'w')
        h.create_dataset('Matrix', data=mtx.cpu().detach().numpy())
        h.create_dataset('PDF', data=pdf)
        h.close()
