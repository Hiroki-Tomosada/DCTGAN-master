import numpy as np
import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from rgb_to_ycbcr import *

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


if __name__ == '__main__':

    paths = glob.glob('/home/tomosada/GOPRO_Large/test/GOPR0384_11_00/sharp/*.png')[:16]
    param_dct = 15
    
    imgs = []
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            array = np.asarray(img, np.float32).transpose([2, 0, 1]) / 255.0
            tensor = torch.as_tensor(np.expand_dims(array, axis=0))
        imgs.append(tensor)
    tensor = torch.cat(imgs, dim=0)

    print(tensor.size())
    tensor_Y = rgb_to_ycbcr(tensor)
    print(tensor_Y.size())
    train_label_dct_0 = torch.abs(dct_2d(tensor_Y[:, 0, :, :]).unsqueeze(1))
    print(train_label_dct_0)
    print(train_label_dct_0.size())
    train_label_dct_1 = train_label_dct_0 > param_dct
    print(train_label_dct_1.size())


    torchvision.utils.save_image(torch.clamp(train_label_dct_0[0], min=0.0, max=1.0), "dct.png")
    