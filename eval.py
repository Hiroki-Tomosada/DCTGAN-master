from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

import scipy.misc
import scipy.ndimage
from skimage.measure import compare_ssim, compare_psnr

import matplotlib
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

mode = 'PSNR_SSIM'
#mode = 'Kernel_size'
#mode = 'Kernel_dataset_eval'

save_path = "/home/tomosada/data/image/Kernel_histgram"

psnr_mode = 1
# GOPRO
#paths_GT = sorted(glob(os.path.join('/home/tomosada/GOPRO_Large/test/*/sharp/*.*')))
# DVD
#paths_GT = sorted(glob(os.path.join('/home/tomosada/DVD_3840FPS_AVG_3-21/test/sharp/*/*.png')))
# NFS
#paths_GT = sorted(glob(os.path.join('/home/tomosada/NFS_3840FPS_AVG_3-21/test/sharp/*/*.png')))

# HIDE

f = open(os.path.join('/home/tomosada/HIDE_dataset', 'test.txt'))
paths_GT = f.read().split()
f.close()
os.chdir('/home/tomosada/HIDE_dataset/GT')

# Kernel
#paths_GT = '/home/tomosada/B100/blur_dataset1/kernel_size.npy'
#paths_GT = '/home/tomosada/B100/blur_dataset2/kernel_size.npy'
#paths_GT = '/home/tomosada/B100/blur_dataset3/kernel_size.npy'
#paths_GT = '/home/tomosada/B100/blur_dataset4/kernel_size.npy'
#paths_GT = '/home/tomosada/B100/blur_dataset5/kernel_size.npy'
#paths_GT = '/home/tomosada/B100/blur_dataset7/kernel_size.npy'

compairs = [
    # GOPRO
    #'/home/tomosada/GOPRO_Large/test/*/blur/*.*', # PSNR: 25.640137742273215 SSIM: 0.858002416602614
    
    # DVD
    #'/home/tomosada/DVD_3840FPS_AVG_3-21/test/blur/*/*.png', # PSNR: 26.973936433792208 SSIM: 0.8462365437638546
    
    # NFS
    #'/home/tomosada/NFS_3840FPS_AVG_3-21/test/blur/*/*.png', # PSNR: 32.09432239348632 SSIM: 0.9426199164576592

    # HIDE
    #'/home/tomosada/HIDE_dataset/test/*/*.png', # PSNR: 23.952992634296983 SSRM: 0.8298751578832886
]

dataset_paths = [
    {"path": "/home/tomosada/GOPRO_kernel/Kernel-Prediction-Normal-3/*/*.npy", "name": "GOPRO", "path_mode": "single"},
    {"path": "/home/tomosada/DVD_kernel/Kernel-Prediction-Normal-3/*/*.npy", "name": "DVD", "path_mode": "single"},
    {"path": "/home/tomosada/NFS_kernel/Kernel-Prediction-Normal-3/*/*.npy", "name": "NFS", "path_mode": "single"},
    {"path": "/home/tomosada/HIDE_kernel/Kernel-Prediction-Normal-3/*.npy", "name": "HIDE", "path_mode": "single"},
    {"path": "/home/tomosada/Kernel_dataset/Adjust6/output_*.pickle", "name": "Adjust6", "path_mode": "multi"},
    {"path": "", "name": "All", "path_mode": "all"},
]

def psnr(img1, img2, mode):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if mode == 0:
        PIXEL_MAX = 255.0
    elif mode == 1:
        PIXEL_MAX = 1
    return 10 * math.log10(PIXEL_MAX * PIXEL_MAX / mse)

def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            size = pickle.load(f)
        return size
    else:
        return []

keys = []
values = []
zeros = []
for i in range(len(compairs)):
    keys.append("Model%d" %(i+1))
    values.append(np.zeros(len(paths_GT)))
    zeros.append(0)

psnrs = dict(zip(keys,values))
psnr_sum = dict(zip(keys, zeros))

ssims = dict(zip(keys,values))
ssim_sum = dict(zip(keys, zeros))

error_sum = dict(zip(keys, zeros))
accuracy_sum = dict(zip(keys, zeros))

fig, ax = plt.subplots()
index = np.arange(0, len(paths_GT))
ax.set_xlabel('t')
ax.set_ylabel('PSNR')
ax.set_title('Title')

if mode == 'PSNR_SSIM':

    print(' [*] Calcurating PSNR, SSIM ...')

    for k, compair in enumerate(compairs):
        print(' [*] Model%d [%s]' %(k+1, compair))
        img_paths = sorted(glob(os.path.join(compair)))

        if len(img_paths) != len(paths_GT):
            print(" [!] You need to test more images")
            print(" [!] Number of Images", len(img_paths), len(paths_GT))
            continue

        for i, path in enumerate(paths_GT):
            GT = cv2.imread(path)
            img = cv2.imread(img_paths[i])

            if psnr_mode == 1:
                GT = GT / 255.
                img = img / 255.

            psnr_i = psnr(GT, img, psnr_mode)
            ssim_i = compare_ssim(GT, img, multichannel=True)

            psnrs["Model%d" %(k+1)][i] = psnr_i
            psnr_sum["Model%d" %(k+1)] = psnr_sum["Model%d" %(k+1)] + psnr_i
            
            ssims["Model%d" %(k+1)][i] = ssim_i
            ssim_sum["Model%d" %(k+1)] = ssim_sum["Model%d" %(k+1)] + ssim_i
            
            print('[%d] %s %s PSNR: %f SSIM: %f' %(i+1, os.path.basename(path), os.path.basename(img_paths[i]), psnr_i, ssim_i))

            if os.path.basename(path) != os.path.basename(img_paths[i]):
                print(" [!] Warning! Name of the file is not same!")
        
        l1 = "Model%d" %(k+1)
        ax.plot(index, psnrs["Model%d" %(k+1)], label=l1)
        
        
        psnr_sum["Model%d" %(k+1)] = psnr_sum["Model%d" %(k+1)] / len(paths_GT)
        ssim_sum["Model%d" %(k+1)] = ssim_sum["Model%d" %(k+1)] / len(paths_GT)

        print('[%d] PSNR: %f SSIM: %f' %(k+1, psnr_sum["Model%d" %(k+1)], ssim_sum["Model%d" %(k+1)]))
        
    ax.legend(loc=0)
    fig.tight_layout()
    plt.savefig('result.png')


    print(psnr_sum)
    print(ssim_sum)

elif mode == 'Kernel_size':
    print(' [*] Calcurating Acculacy ...')

    GT = np.load(paths_GT)

    for k, compair in enumerate(compairs):
        print(' [*] Model%d [%s]' %(k+1, compair))
        
        imply = np.load(compair)

        if len(GT) != len(imply):
            print(' [!] Data size is not same', len(GT), len(imply))
            continue

        for i, number in enumerate(GT):
            print(' [{0:010d}] GT: {1}, Imply: {2}'.format(i+1, number, imply[i]))
            if int(number) == int(imply[i]):
                accuracy_sum["Model%d" %(k+1)] += 1
            
            error_sum["Model%d" %(k+1)] += abs(int(number) - int(imply[i]))
        
        accuracy_sum["Model%d" %(k+1)] = accuracy_sum["Model%d" %(k+1)] / len(GT) * 100
        error_sum["Model%d" %(k+1)] = error_sum["Model%d" %(k+1)] / len(GT)
        
        print('[%d] Accuracy: %f Error: %f' %(k+1, accuracy_sum["Model%d" %(k+1)], error_sum["Model%d" %(k+1)]))
        
    print(accuracy_sum)
    print(error_sum)

elif mode == 'Kernel_dataset_eval':
    for n, dataset_path in enumerate(dataset_paths):
        paths = sorted(glob(dataset_path['path']))
        
        master_size = []
        if dataset_path['path_mode'] == "multi":
            for i, path in enumerate(paths):
                size_paths = load_pickle(os.path.join(path))

                for j, size_path in enumerate(size_paths):
                    imply = np.load(size_path)

                    if i == 0 and j == 0:
                        master_size = imply
                    else:
                        master_size = np.append(master_size, imply)

        elif dataset_path['path_mode'] == "single":
            for i, path in enumerate(paths):
                imply = np.load(path)

                if i == 0:
                    master_size = imply
                else:
                    master_size = np.append(master_size, imply)
        
        elif dataset_path['path_mode'] == "all":
            paths = [
                "/home/tomosada/GOPRO_kernel/Kernel-Prediction-Normal-3/*/*.npy",
                "/home/tomosada/DVD_kernel/Kernel-Prediction-Normal-3/*/*.npy",
                "/home/tomosada/NFS_kernel/Kernel-Prediction-Normal-3/*/*.npy",
                "/home/tomosada/HIDE_kernel/Kernel-Prediction-Normal-3/*.npy"
            ]

            for i, path in enumerate(paths):
                size_paths = sorted(glob(path))

                for j, size_path in enumerate(size_paths):
                    imply = np.load(size_path)

                    if i == 0 and j == 0:
                        master_size = imply
                    else:
                        master_size = np.append(master_size, imply)

        print("[{0:04d}] {1} {2:010d}".format(n, dataset_path['name'], len(master_size)))

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(master_size, bins=300)
        ax.set_title(dataset_path['name'])
        ax.set_xlim(0, 30)
        ax.set_xlabel('size')
        ax.set_ylabel('number')
        
        fig.savefig(os.path.join(save_path, "result_" + dataset_path['name'] + ".png"))

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in' 
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(master_size, bins=300)
        ax.set_xlim(0, 30)
        ax.set_xlabel('Kernel size')
        ax.set_ylabel('number')
        
        fig.savefig(os.path.join(save_path, "paper_" + dataset_path['name'] + ".png"))