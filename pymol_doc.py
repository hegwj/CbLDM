import os
from tqdm import tqdm


def count_files_in_dir(dir):
    return len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])


file_path = ('E:/useful/Project/Pro_Diff/DiffStruc_modi/models/2/pred'
             '/graph_HCP_Size1_4_Size2_5_Size3_5_atom_Zn_lc1_2.42_lc2_3.95186')
save_path = ('E:/useful/Project/Pro_Diff/DiffStruc_modi/models/2/pred'
             '/diffuse_HCP_png')

num_max = count_files_in_dir(file_path)

with open(f'./pymol_doc0.txt', 'w') as file:
    file.write('set sphere_scale, 0.5\n')

for k in range((num_max // 100) + 1):
    with open(f'./pymol_doc{k}.txt', 'a') as file:
        if k != (num_max // 100):
            for i in tqdm(range(100)):
                file.write(f'load {file_path}/{i + k * 100}.xyz\n')
                file.write(f'zoom {i + k * 100}, 25\n')
                file.write('ray 512, 512\n')
                file.write(f'save {save_path}/{i + k * 100}.png\n')
                file.write('delete all\n')
        else:
            for i in tqdm(range((num_max - k * 100))):
                file.write(f'load {file_path}/{i + k * 100}.xyz\n')
                file.write(f'zoom {i + k * 100}, 25\n')
                file.write('ray 512, 512\n')
                file.write(f'save {save_path}/{i + k * 100}.png\n')
                file.write('delete all\n')
