from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision

import glob

def make_blur(args, tensor, kernel):
    with torch.no_grad():
        for i in range(args.batch_size):
            tensor[i,:,:,:] = F.conv2d(tensor[i,:,:,:].unsqueeze(0), kernel[i,:,:,0].unsqueeze(0).expand(3, args.kernel_flame, args.kernel_flame).unsqueeze(0), padding=int(args.kernel_flame//2)).unsqueeze(1) # torch.Size([8, 1, 256, 256]) torch.Size([1, 1, kernel_size, kernel_size])
    return tensor
