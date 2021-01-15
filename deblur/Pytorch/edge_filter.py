from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision

import glob

def Laplacian_filter(tensor, kernel, gray_k):
    with torch.no_grad():
        gray = torch.sum(tensor * gray_k, dim=1, keepdim=True)
        edge_image = F.conv2d(gray, kernel, padding=1)
        
    return edge_image

