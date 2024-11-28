import h5py
import torch
import numpy as np

from scipy import linalg
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm
from modules.vae_ldm import Net_ldm as Net
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

def save_xyz_file(save_dir, cords, file_name):

    cords = np.array(cords)
    cords[:, 0] -= cords[0, 0]
    cords[:, 1] -= cords[0, 1]
    cords[:, 2] -= cords[0, 2]
    these_cords = []

    for count, xyz in enumerate(cords):
        if count == 0:
            these_cords.append(['{:d}'.format(len(cords))])
            these_cords.append([''])

        these_cords.append(['W   {:.4f}   {:.4f}   {:.4f}'.format(xyz[0], xyz[1], xyz[2])])

    np.savetxt(save_dir + '/{}.xyz'.format(file_name), these_cords, fmt='%s')
    return None


def distmatrix2xyz_laplace(D, t):
    n = D.shape[0]
    M = np.diag(np.sum(D + D.T, axis=0) / 2)

    L = M - (D + D.T) / 2
    k, V = linalg.eig(L, M)

    sorted_seq = np.argsort(k)
    V = V[:, sorted_seq]

    Y = V[:, 1:4]
    Y = Y.flatten()

    target = lambda X: loss_fun(X, t, D)
    jacobian = lambda X: myjac(X, t, D)

    X = minimize(target, Y, jac=jacobian,
                 method='trust-constr', tol=1e-8, options={'maxiter': 200})

    X = X.x.reshape((n, 3))

    return X

def myfun(X, t, W):
    D = torch.norm(X[:, None] - X, dim=2, p=2) ** 2
    T = torch.exp(-D / t)
    F = torch.sum((T - W) ** 2)

    return F

def loss_fun(X, t, W):
    n = X.shape[0] // 3
    X = X.reshape(n, 3)

    X = torch.tensor(X, requires_grad=False, dtype=torch.float64)
    W = torch.tensor(W, requires_grad=False, dtype=torch.float64)
    F = myfun(X, t, W)

    F = F.detach().numpy()

    return F

def myjac(X, t, W):
    n = X.shape[0] // 3
    X = X.reshape(n, 3)

    X = torch.tensor(X, requires_grad=True, dtype=torch.float64)
    W = torch.tensor(W, requires_grad=False, dtype=torch.float64)
    F = myfun(X, t, W)
    F.backward()

    grad_X = X.grad.flatten()
    grad_X = grad_X.detach().numpy()

    return grad_X

def get_xyz(m_pred, t_val=100, method='laplace'):
    m = torch.zeros([256, 256])

    m[0:128, 0:128] = m_pred[0, 0, :, :]
    m[0:128, 128:256] = m_pred[0, 1, :, :]
    m[128:256, 0:128] = m_pred[0, 2, :, :]
    m[128:256, 128:256] = m_pred[0, 3, :, :]

    m_np = m.cpu().detach().numpy()
    m_max = np.max(m_np, axis=0)
    b = m_max < np.exp(- 8 ** 2 / t_val)
    count = len(b[b[:] == 0])
    m = m[:count, :count]  # m = m[:count, :count]

    if method == 'laplace':
        m = m.cpu().detach().numpy()

        xyz = distmatrix2xyz_laplace(m, t_val)
        xyz = torch.from_numpy(xyz)

        m = torch.norm(xyz[:, None] - xyz, dim=2, p=2)
        m = m.cpu().detach().numpy()
        print(m)
        print(np.max(m))

    elif method == 'mds':
        m = (m + m.T) / 2
        m = (- t_val * torch.log(m)) ** .5
        m = m.cpu().detach().numpy()

        for i in range(count):
            m[i, i] = 0
    mds = MDS(n_components=3, dissimilarity='precomputed', normalized_stress='auto', random_state=42)
    xyz = mds.fit_transform(m)
    print(j + 1)
    print(count)
    print('-------------------------------')

    return xyz


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num = 24
    rep = 1
    print_or_not = True
    adgp_method = 'laplace'  # 'laplace' or 'mds'

    path = 'D:\\01.张志扬\\01.学习\\研究生\\代码文件\\DiffStruc_modi\\model_val_loss=0.13734.ckpt'

    # net = Net(beta_schedule='modified')
    net = Net(beta_schedule='cosine')
    model = Net(beta_schedule='cosine').load_from_checkpoint(path).eval().to(device)

    # file_path = f'E:/useful/Project/AnyDataset/data_train/'
    # file_path = f'E:/useful/Project/AnyDataset/CbLDM2/BCC888/'
    # file_path = f'E:/useful/Project/AnyDataset/CbLDM2/SC333BCC355/'
    csv_file_path = 'your_filename_csv_path'
    df = pd.read_csv(csv_file_path)

    h5_filenames = df['filename'].tolist()
    #file_name = 'C60'

    # graph_SC_h_1_k_1_l_1_atom_Ni_lc_2.3
    # graph_Oct_length_2_atom_Ni_lc_2.3
    # graph_Ico_shell_2_atom_Ni_lc_2.3
    # graph_HCP_Size1_4_Size2_5_Size3_5_atom_Zn_lc1_2.42_lc2_3.95186
    # graph_Oct_length_7_atom_W_lc_2.6
    # graph_FCC_h_2_k_4_l_10_atom_W_lc_2.6

    # graph_BCC_h_8_k_8_l_8_atom_Zr_lc_2.9

    # graph_BCC_h_3_k_5_l_5_atom_Hg_lc_2.78
    # graph_SC_h_3_k_3_l_3_atom_W_lc_2.6

    # JQ_S1
    # Au144pMBA
    # Au144PET-100
    # Au144SC6_100K_APS
    # C60
    # A144
    file_path = 'your_data_path'
    for filename in tqdm(h5_filenames):
        file = h5py.File(file_path + '\\' + filename, 'r')
        pdf = file['PDF'][:]
        pdf = pdf[np.newaxis, :]
        pdf = torch.tensor(pdf, dtype=torch.float).to(device)
    # pdf = torch.tensor(pdf_m / np.max(np.abs(pdf_m)), dtype=torch.float).to(device)

        for j in range(rep):
            m_pred = model(pdf, pred=True, prt=900)
            print(m_pred.shape)
            xyz = get_xyz(m_pred, 260, method=adgp_method)
        save_path = 'save_path'
        save_xyz_file(save_path, xyz, filename + '_p')
