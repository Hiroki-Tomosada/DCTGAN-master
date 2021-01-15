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

def gopro_image_list(dataset_path):
    image_list = []
    paths_gt = sorted(glob.glob(os.path.join(dataset_path, '*/sharp/*.png')))
    paths_blur = sorted(glob.glob(os.path.join(dataset_path, '*/blur/*.png')))
    for i, path in enumerate(paths_gt):
        image_list.append((paths_blur[i], path))
    return image_list


def collect_gopro_image_list(dataset_path):
    image_list = []
    for parent_dir in os.listdir(dataset_path):
        blur_directory = os.path.join(dataset_path, parent_dir, 'blur')
        sharp_directory = os.path.join(dataset_path, parent_dir, 'sharp')
        for image_name in sorted(os.listdir(blur_directory)):
            blur_image_path = os.path.join(blur_directory, image_name)
            sharp_image_path = os.path.join(sharp_directory, image_name)
            if os.path.exists(blur_image_path) and os.path.exists(sharp_image_path):
                image_list.append((blur_image_path, sharp_image_path))
    return image_list


class GOPRO_Dataset(Dataset):
    def __init__(self, args, normalize=False):
        self.dataset_path = os.path.expanduser(args.dataset_train)
        self.image_list = gopro_image_list(self.dataset_path)
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


class TestDataset(Dataset):
    def __init__(self, data_folder_test):
        self.tensor_setup = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.files_blurred = sorted(glob.glob(data_folder_test))

    def __getitem__(self, index):
        filePath_blurred = self.files_blurred[index % len(self.files_blurred)]
        data = np.array(Image.open(filePath_blurred), 'f') / 255.
        return {'Input': self.tensor_setup(data), 'Filename': filePath_blurred}

    def __len__(self):
        return len(self.files_blurred)

if __name__ == '__main__':
    dataset_path = "/home/tomosada/GOPRO_Large/train"
    dataset_path = os.path.expanduser(dataset_path)
    image_list = collect_gopro_image_list(dataset_path)
    print(image_list[100])

    image_list = gopro_image_list(dataset_path)
    print(image_list[100])