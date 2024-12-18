import h5py
import numpy as np
from tqdm import tqdm
import os


def trans_data(root):
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
    path = 'your_pdf_path'
    path_save = 'your_save_path'

    for i in range(1):
        for file in tqdm(os.listdir(path)):
            pdf = trans_data(path + file)
            h = h5py.File(path_save + '_' + file, 'w')
            h.create_dataset('PDF', data=pdf)
            h.close()
