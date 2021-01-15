#!/opt/anaconda3/bin/python
import os, glob, re, sys
import random
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kernel_generation import *

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

def mscoco_image_list(dataset_path):
    image_list = []
    paths_gt = sorted(glob.glob(os.path.join(dataset_path, '*.jpg')))
    paths_blur = sorted(glob.glob(os.path.join(dataset_path, '*.jpg')))
    for i, path in enumerate(paths_gt):
        image_list.append((paths_blur[i], path))
    return image_list

class MSCOCO_Dataset(Dataset):
    def __init__(self, args, normalize=False):
        self.args = args
        self.dataset_path = os.path.expanduser(self.args.dataset_train)
        self.image_list = mscoco_image_list(self.dataset_path)
        self.patch_size = args.image_size
        self.linear_rate = args.linear_rate

        self.tensor_setup = transforms.Compose(
            [
                transforms.Normalize(mean, std)
            ]
        )
    
    def __getitem__(self, n):
        blur_image_path, sharp_image_path = self.image_list[n]
        sharp_image = read_image(sharp_image_path)
        sharp_img_np = cv2.imread(sharp_image_path)
        height, width = sharp_image.shape[-2:]
        origin_y = random.randint(0, height - self.patch_size - 1)
        origin_x = random.randint(0, width - self.patch_size - 1)
        sharp_img_np = sharp_img_np[origin_y : origin_y + self.patch_size, origin_x : origin_x + self.patch_size, :]
        sharp_image = sharp_image[..., origin_y : origin_y + self.patch_size, origin_x : origin_x + self.patch_size]
        
        if random.random() < self.linear_rate:
            kernel, kernel_img, kernel_size = get_random_linear_kernel(self.args)
        else:
            kernel, kernel_img, kernel_size = get_random_non_linear_kernel(self.args)

        blur_np = cv2.filter2D(sharp_img_np, -1, kernel)
        blur_image = torchvision.transforms.functional.to_tensor(Image.fromarray(blur_np))
        kernel_image = torchvision.transforms.functional.to_tensor(Image.fromarray(kernel_img))
        
        return {'Input': blur_image, 'GrandTruth': sharp_image, 'Kernel': kernel_image, 'Kernel_size': kernel_size}
        
    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    dataset_path = "/home/tomosada/MSCOCO/train2017"

    image_list = mscoco_image_list(dataset_path)
    print(image_list[100])

    size_min = 10000
    count = 0
    paths = sorted(glob.glob(dataset_path + "/*.jpg"))
    for path in paths:
        img = Image.open(path)
        #print(img.size)
        #size_min = min(size_min, img.size[1])
        if min(img.size) < 256:
            print('ERROR')
            count = count + 1
        elif img.mode == 'RGB':
            print('ERROR Gray')
            count = count + 1
    print(count)
    print(len(paths))
    
    #print(size_min)