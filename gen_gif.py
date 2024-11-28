import imageio
import os
import imageio
import cv2


def count_files_in_dir(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def pic2gif(gif_path, gif_name, pic_path, t, scale):
    num = count_files_in_dir(pic_path)
    gif_images = []
    for i in range(num):
        img = imageio.v3.imread(f'{pic_path}/{i}.png')
        h = img.shape[0]
        w = img.shape[1]
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        gif_images.append(img)
    gif_file = gif_path + '/' + gif_name
    imageio.mimsave(gif_file,gif_images,'mp4', fps=30)


if __name__ == '__main__':
    pic_path = './models/2/pred/diffuse_HCP_png'
    gif_name = 'diffuse_HCP.mp4'
    gif_path = './gif'
    t = 0.005
    scale = 1
    pic2gif(gif_path, gif_name, pic_path, t, scale)
