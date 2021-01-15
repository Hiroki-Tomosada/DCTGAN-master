#!/opt/anaconda3/bin/python
import os, glob, re
import random
import numpy as np
import PIL
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

def HIDE_image_list(dataset_path):
    f = open(os.path.join(dataset_path, 'train.txt'))
    image_list = f.read().split()
    f.close()
    '''
    paths_gt = sorted(glob.glob(os.path.join(dataset_path, '*/sharp/*.png')))
    paths_blur = sorted(glob.glob(os.path.join(dataset_path, '*/blur/*.png')))
    for i, path in enumerate(paths_gt):
        image_list.append((paths_blur[i], path))
    '''
    return image_list

class HIDE_Dataset(Dataset):
    def __init__(self, args, normalize=False):
        self.dataset_path = os.path.expanduser(args.dataset_train)
        self.image_list = HIDE_image_list(self.dataset_path)
        self.patch_size = args.image_size

        self.tensor_setup = transforms.Compose(
            [
                transforms.Normalize(mean, std)
            ]
        )
    
    def __getitem__(self, n):
        image_path = self.image_list[n]
        blur_image = read_image(os.path.join(self.dataset_path, 'train', image_path))
        sharp_image = read_image(os.path.join(self.dataset_path, 'GT', image_path))
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
    image_list = collect_gopro_image_list(dataset_path)
    print(image_list[100])

    image_list = gopro_image_list(dataset_path)
    print(image_list[100])