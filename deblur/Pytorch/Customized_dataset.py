#!/opt/anaconda3/bin/python
import os, glob, re
import random
import numpy as np
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def get_image_list(dataset_path):
    image_list = []
    for ext in ('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'):
        image_list += sorted(glob.glob(dataset_path + '/*.' + ext))
    return image_list


def read_image(image_path):
    img = Image.open(image_path)
    return torchvision.transforms.functional.to_tensor(img)

def Customized_image_list(dataset_path):
    image_list = []
    paths_gt = []
    paths_blur = []

    if os.path.exists(os.path.join(dataset_path, "Kernel_dataset", "Customized", "output.pickle")):
        with open(os.path.join(dataset_path, "Kernel_dataset", "Customized", "output.pickle"), 'rb') as f:
            min_size, max_size = pickle.load(f)
    else:
        print(' [!] Please indicate correct dataset path')
    
    for i in range(min_size, max_size + 1):
        if os.path.exists(os.path.join(dataset_path, "Kernel_dataset", "Customized", "output_" + str(i) + ".pickle")):
            with open(os.path.join(dataset_path, "Kernel_dataset", "Customized", "output_" + str(i) + ".pickle"), 'rb') as f:
                paths = pickle.load(f)

            for path in enumerate(paths):
                if 'GOPRO_' in path[1]:
                    paths_gt.append(os.path.join(dataset_path, 'GOPRO_Large', 'train', os.path.basename(os.path.dirname(path[1])), 'sharp', os.path.basename(os.path.splitext(path[1])[0]) + '.png'))
                    paths_blur.append(os.path.join(dataset_path, 'GOPRO_Large', 'train', os.path.basename(os.path.dirname(path[1])), 'blur', os.path.basename(os.path.splitext(path[1])[0]) + '.png'))
                elif 'DVD_' in path[1]:
                    paths_gt.append(os.path.join(dataset_path, 'DVD_3840FPS_AVG_3-21', 'train', 'sharp', os.path.basename(os.path.dirname(path[1])), os.path.basename(os.path.splitext(path[1])[0]) + '.png'))
                    paths_blur.append(os.path.join(dataset_path, 'DVD_3840FPS_AVG_3-21', 'train', 'blur', os.path.basename(os.path.dirname(path[1])), os.path.basename(os.path.splitext(path[1])[0]) + '.png'))
                elif 'NFS_' in path[1]:
                    paths_gt.append(os.path.join(dataset_path, 'NFS_3840FPS_AVG_3-21', 'train', 'sharp', os.path.basename(os.path.dirname(path[1])), os.path.basename(os.path.splitext(path[1])[0]) + '.png'))
                    paths_blur.append(os.path.join(dataset_path, 'NFS_3840FPS_AVG_3-21', 'train', 'blur', os.path.basename(os.path.dirname(path[1])), os.path.basename(os.path.splitext(path[1])[0]) + '.png'))
                elif 'HIDE_' in path[1]:
                    paths_gt.append(os.path.join(dataset_path, 'HIDE_dataset', 'GT', os.path.basename(os.path.splitext(path[1])[0]) + '.png'))
                    paths_blur.append(os.path.join(dataset_path, 'HIDE_dataset', 'train', os.path.basename(os.path.splitext(path[1])[0]) + '.png'))
    
    for i, path in enumerate(paths_gt):
        image_list.append((paths_blur[i], path))
    return image_list

class Customized_Dataset(Dataset):
    def __init__(self, args, normalize=False):
        self.dataset_path = os.path.expanduser(args.dataset_train)
        self.image_list = Customized_image_list(self.dataset_path)
        self.patch_size = args.image_size

        self.tensor_setup = transforms.Compose(
            [
                transforms.Normalize(mean, std)
            ]
        )
    
    def __getitem__(self, n):
        blur_image_path, sharp_image_path = self.image_list[n]
        blur_image = read_image(blur_image_path)
        sharp_image = read_image(sharp_image_path)
        height, width = blur_image.shape[-2:]
        origin_y = random.randint(0, height - self.patch_size - 1)
        origin_x = random.randint(0, width - self.patch_size - 1)
        blur_image = blur_image[..., origin_y : origin_y + self.patch_size,
                                origin_x : origin_x + self.patch_size]
        sharp_image = sharp_image[..., origin_y : origin_y + self.patch_size,
                                  origin_x : origin_x + self.patch_size]
        return {'Input': blur_image, 'GrandTruth': sharp_image}
        
    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    dataset_path = "/home/tomosada/GOPRO_Large/train"
    dataset_path = os.path.expanduser(dataset_path)
    
    image_list = DVD_image_list(dataset_path)
    print(image_list[100])