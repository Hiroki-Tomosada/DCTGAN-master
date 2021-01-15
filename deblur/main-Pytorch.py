import argparse
import os
import numpy as np
import math
import sys
import codecs
import pickle
import tqdm
import datetime
import json

import cv2
from skimage.measure import compare_ssim, compare_psnr
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch

from kernel_generation import *
from Pytorch.make_blur import make_blur

from Pytorch.GOPRO_dataset import GOPRO_Dataset, TestDataset
from Pytorch.MSCOCO_dataset import MSCOCO_Dataset
from Pytorch.HIDE_dataset import HIDE_Dataset
from Pytorch.DVD_dataset import DVD_Dataset
from Pytorch.NFS_dataset import NFS_Dataset
from Pytorch.Customized_dataset import Customized_Dataset
from Pytorch.MSCOCO_kernel_class import MSCOCO_Kernel_Class

from circle import *
from Pytorch.rgb_to_ycbcr import *
from Pytorch.edge_filter import Laplacian_filter

sys.path.append(os.path.join(os.path.dirname(__file__), 'DCTGAN-P'))
from DCTGAN import DCTGAN_G, DCTGAN_D1, DCTGAN_D2, VGG19
from dct import dct_2d
sys.path.append(os.path.join(os.path.dirname(__file__), 'Kernel_prediction-P'))
from Kernel_prediction import Kernel_prediction

version = 'Version 5.8'
print(" [*] Pytorch Controller [%s] powered by Tomosada" % version)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id",type=int,default=0)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument("--model",type=str,default='DCTGAN', help='model architecture')
parser.add_argument("--train_name",type=str,default="Normal")
parser.add_argument("--param_name",type=str,default="1")
parser.add_argument("--data_folder",type=str,default="")
parser.add_argument("--output_name",type=str,default="output")
parser.add_argument("--log_name",type=str,default="logs")
parser.add_argument("--checkpoint",type=str,default="checkpoint")
parser.add_argument("--log_freq",default=100,type=int)
parser.add_argument("--save_freq",default=100000,type=int)
parser.add_argument("--load_G",type=str,default="generator")
parser.add_argument("--load_D1",type=str,default="discriminator1")
parser.add_argument("--load_D2",type=str,default="discriminator2")
parser.add_argument("--num_display", type=int,default=4)
parser.add_argument("--scale", type=int,default=1)

parser.add_argument('--kernel_mode',type=str,default="No-Kernel") # Kernel-Generation, Estimate-Kernel-size, Estimate-Kernel-class
parser.add_argument('--linear_rate',default=0,type=float)
parser.add_argument("--kernel_flame",default=61,type=int)
parser.add_argument("--kernel_max",default=51,type=int)
parser.add_argument("--kernel_min",default=11,type=int)
parser.add_argument("--all_sigma",default=1,type=int)
parser.add_argument("--sigma_min",default=1,type=int)
parser.add_argument("--sigma_max",default=20,type=int)
parser.add_argument("--add_angle_min",default=0,type=int)
parser.add_argument("--add_angle_max",default=0,type=int)
parser.add_argument("--cache",default=10000,type=int)
parser.add_argument("--kernel_size_class",default=3,type=int)

parser.add_argument("--epoch",default=250,type=int)
parser.add_argument("--pretrain", default=4, type=int, help="pretrained data")
parser.add_argument('--step_mode', action='store_true')
parser.add_argument("--step",default=500000,type=int)
parser.add_argument("--pretrain_step",default=13000,type=int)
parser.add_argument("--learning_rate",default=1e-4,type=float)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--batch_size",default=8,type=int)

parser.add_argument("--param_MSE",default=1,type=float)
parser.add_argument("--param_l1",default=1,type=float)
parser.add_argument("--param_Laplacian",default=1,type=float)
parser.add_argument("--param_DCT",default=1,type=float)
parser.add_argument("--param_VGG",default=0.1,type=float)
parser.add_argument("--param_Cross_Entropy",default=1,type=float)

parser.add_argument("--param2_MSE",default=1,type=float)
parser.add_argument("--param2_l1",default=12,type=float)
parser.add_argument("--param2_Laplacian",default=1,type=float)
parser.add_argument("--param2_DCT",default=12,type=float)
parser.add_argument("--param2_VGG",default=0.05,type=float)
parser.add_argument("--param2_D1",default=2,type=float)
parser.add_argument("--param2_D2",default=4,type=float)

parser.add_argument('--dct_binary', action='store_true')
parser.add_argument("--dct_threshold",default=15,type=float)
parser.add_argument('--round_mask', action='store_true')
parser.add_argument("--mask_threshold",default=96,type=int)

parser.add_argument("--dataset",type=str,default="Customized")
parser.add_argument("--dataset_train",type=str,default="")
parser.add_argument("--image_size",default=256,type=int)
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")

parser.add_argument('--test', action='store_true')
parser.add_argument('--patch', action='store_true')
parser.add_argument("--test_dataset",default="") 
parser.add_argument("--result",default='')
args = parser.parse_args()

train_name = args.model + '-' + args.train_name + '-' + args.param_name

start = datetime.datetime.now()
start_at = start.strftime('%Y/%m/%d %H:%M:%S')

os.makedirs(os.path.join(args.data_folder, args.model + '-P'), exist_ok=True)
os.chdir(os.path.join(args.data_folder, args.model + '-P'))
print(' [*] Current Directory', os.getcwd())

archs = {
    'DCTGAN': {
        'Normal': {
            'Generator': DCTGAN_G, 'Discriminator1': DCTGAN_D1, 'Discriminator2': None, 'FeatureExtractor': VGG19, 
            'Loss_CNN': ['MSE'], 'Loss_GAN': ['L1', 'DCT', 'VGG19']},
        'Normal-LGAN': {
            'Generator': DCTGAN_G, 'Discriminator1': None, 'Discriminator2': None, 'FeatureExtractor': VGG19, 
            'Loss_CNN': ['MSE'], 'Loss_GAN': ['L1', 'DCT', 'VGG19']},
        'Normal-LDCT': {
            'Generator': DCTGAN_G, 'Discriminator1': DCTGAN_D1, 'Discriminator2': None, 'FeatureExtractor': VGG19, 
            'Loss_CNN': ['MSE'], 'Loss_GAN': ['L1', 'VGG19']},
        'Normal-LVGG': {
            'Generator': DCTGAN_G, 'Discriminator1': DCTGAN_D1, 'Discriminator2': None, 'FeatureExtractor': None, 
            'Loss_CNN': ['MSE'], 'Loss_GAN': ['L1', 'DCT']},
    },
    'Kernel-Prediction': {
        'Normal': {
            'Generator': Kernel_prediction, 'Discriminator1': None, 'Discriminator2': None, 'FeatureExtractor': None, 
            'Loss_CNN': ['L1'], 'Loss_GAN': []},
    },
}

load_dataset = {
    'GOPRO': GOPRO_Dataset, 'MSCOCO': MSCOCO_Dataset, 'HIDE': HIDE_Dataset, 'DVD': DVD_Dataset, 'NFS': NFS_Dataset, 'Customized': Customized_Dataset,
}

def train():
    print(' [*] Train Session')

    if torch.cuda.is_available(): 
        device = 'cuda'
        if not args.multi_gpu:
            torch.cuda.set_device(args.gpu_id)
            torch.backends.cudnn.benchmark=True
    else:
        device = 'cpu'

    print(' [*] Building model', train_name)

    if not archs[args.model][args.train_name]['Generator'] == None:
        Generator = archs[args.model][args.train_name]['Generator']()
        Generator.to(device)
        if device == 'cuda' and args.multi_gpu:
            Generator = torch.nn.DataParallel(Generator)
            torch.backends.cudnn.benchmark=True
        Generator.train()
    else:
        print(' [!] Your Architecture has no Generator')
        exit()
    if not archs[args.model][args.train_name]['Discriminator1'] == None:
        Discriminator_1 = archs[args.model][args.train_name]['Discriminator1']()
        Discriminator_1.to(device)
        if device == 'cuda' and args.multi_gpu:
            Discriminator_1 = torch.nn.DataParallel(Discriminator_1)
        Discriminator_1.train()
    if not archs[args.model][args.train_name]['Discriminator2'] == None:
        Discriminator_2 = archs[args.model][args.train_name]['Discriminator2']()
        Discriminator_2.to(device)
        if device == 'cuda' and args.multi_gpu:
            Discriminator_2 = torch.nn.DataParallel(Discriminator_2)
        Discriminator_2.train()
    if not archs[args.model][args.train_name]['FeatureExtractor'] == None:
        feature_extractor = archs[args.model][args.train_name]['FeatureExtractor']()
        feature_extractor.to(device)
        if device == 'cuda' and args.multi_gpu:
            feature_extractor = torch.nn.DataParallel(feature_extractor)
        feature_extractor.eval()

    criterion_MSE = torch.nn.MSELoss()
    criterion_L1 = torch.nn.L1Loss()
    criterion_BCE = torch.nn.BCELoss()
    criterion_CE = torch.nn.CrossEntropyLoss()

    masking = np.zeros([args.batch_size, 1, 256, 256])
    masking_0 = circle(args.mask_threshold)
    masking_1 = np.expand_dims(np.expand_dims(masking_0, 0), 0)
    for i in range(args.batch_size):
        masking[i, :, :, :] = masking_1
    masking = torch.Tensor(masking).to(device)

    if 'Laplacian' in archs[args.model][args.train_name]['Loss_CNN'] or 'Laplacian' in archs[args.model][args.train_name]['Loss_GAN']:
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
        edge_kernel = torch.as_tensor(edge_kernel.reshape(1, 1, 3, 3)).to(device)
        gray_kernel = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1)
        gray_k = torch.as_tensor(gray_kernel).to(device)
    
    print(" [*] Reading checkpoints...")
    os.makedirs(args.checkpoint, exist_ok=True)

    if not os.path.exists(os.path.join('.', args.checkpoint, train_name)):
        os.makedirs(os.path.join('.', args.checkpoint, train_name))
    
    if os.path.exists(os.path.join(args.checkpoint, train_name, 'status.pickle')):
        print(" [*] Load SUCCESS", train_name)
        with open(os.path.join(args.checkpoint, train_name, 'status.pickle'), 'rb') as f:
            counter, global_epoch, global_step = pickle.load(f)
        if ((not args.step_mode) and global_epoch >= args.epoch) or (args.step_mode and global_step >= args.step):
            print(' [!] You do not need to train Epoch: {} Global Step: {} '.format(global_epoch, global_step))
            exit()
        print(" [*] Global Epoch:", global_epoch, 'Step:', global_step)
        if ((not args.step_mode) and global_epoch > args.pretrain) or (args.step_mode and global_step > args.pretrain_step):
            Generator.load_state_dict(torch.load(os.path.join(args.checkpoint, train_name, args.load_G + "_" + str(global_step) + ".pth")))
            if os.path.exists(os.path.join(args.checkpoint, train_name, args.load_D1 + "_" + str(global_step) + ".pth")):
                Discriminator_1.load_state_dict(torch.load(os.path.join(args.checkpoint, train_name, args.load_D1 + "_" + str(global_step) + ".pth")))
            if os.path.exists(os.path.join(args.checkpoint, train_name, args.load_D2 + "_" + str(global_step) + ".pth")):
                Discriminator_2.load_state_dict(torch.load(os.path.join(args.checkpoint, train_name, args.load_D2 + "_" + str(global_step) + ".pth")))
            status = 'GAN'
        else:
            Generator.load_state_dict(torch.load(os.path.join(args.checkpoint, train_name, args.load_G + "_" + str(global_step) + ".pth")))
            status = 'CNN'
    else:
        print(" [!] Load failed and Initializing ...")
        counter = 0
        global_epoch = 0
        global_step = 0
    
    counter = counter + 1
    global_epoch = global_epoch + 1
    global_step = global_step + 1
    start_step = global_step

    save_parameter(args, archs, counter)
    
    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
    if not archs[args.model][args.train_name]['Discriminator1'] == None:
        optimizer_D_1 = torch.optim.Adam(Discriminator_1.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
    if not archs[args.model][args.train_name]['Discriminator2'] == None:
        optimizer_D_2 = torch.optim.Adam(Discriminator_2.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))

    valid = Variable(torch.ones((args.batch_size, 1)), requires_grad=False).to(device)
    fake = Variable(torch.zeros((args.batch_size, 1)), requires_grad=False).to(device)

    dataloader = DataLoader(
        load_dataset[args.dataset](args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
    )

    writer = SummaryWriter(log_dir=os.path.join(args.log_name, train_name))
    
    txt3 = []
    txt3.append("-- Logs [%s] --" % version)
    txt3.append("Epoch,Step,G_Loss,D1_Loss,D2_Loss")

    # ----------
    #  Training
    # ----------
    for epoch in range(global_epoch, args.epoch + 1):

        print(train_name, '- Deblur -', 'Epoch: {}'.format(epoch), 'Global Step: {}'.format(global_step))
        for imgs in tqdm.tqdm(dataloader):

            if args.kernel_mode == 'Estimate-Kernel-size':
                train_label = torch.unsqueeze(imgs['Kernel_size'], -1)
            elif args.kernel_mode == 'Estimate-Kernel-class':
                train_label = torch.unsqueeze(imgs['Kernel_class'], -1)
            else:
                train_label = imgs['GrandTruth']
            
            train_data = imgs['Input']
            
            train_label_ori = train_label.to(device)
            train_data_ori = train_data.to(device, dtype=torch.float)

            for i in range(1 - args.scale, 1):
                if args.scale != 1:
                    train_label = F.interpolate(train_label_ori, (int(args.image_size*(2**i)), int(args.image_size*(2**i))), mode='nearest')
                    data_scale1 = F.interpolate(train_data_ori, (int(args.image_size*(2**i)), int(args.image_size*(2**i))), mode='nearest')

                    if i == 1 - args.scale:
                        data_scale2 = data_scale1.clone()
                    else:
                        data_scale2 = F.interpolate(g_train_output.detach(), (int(args.image_size*(2**i)), int(args.image_size*(2**i))), mode='nearest')
                    
                    train_data = torch.cat((data_scale1, data_scale2), dim=1)
                else:
                    train_label = train_label_ori
                    train_data = train_data_ori
            
                is_first = True

                if ((not args.step_mode) and epoch <= args.pretrain) or (args.step_mode and global_step <= args.pretrain_step):
                    status = 'CNN'
                    # ------------------
                    #  Train Generators
                    # ------------------
                    optimizer_G.zero_grad()

                    g_train_output = Generator(train_data)

                    if 'L1' in archs[args.model][args.train_name]['Loss_CNN']:
                        loss_l1 = criterion_L1(train_label, g_train_output)
                        loss_G = loss_l1 * args.param_l1 if is_first else loss_G + loss_l1 * args.param_l1
                        is_first = False

                    if 'MSE' in archs[args.model][args.train_name]['Loss_CNN']:
                        loss_MSE = criterion_MSE(train_label, g_train_output)
                        loss_G = loss_MSE * args.param_MSE if is_first else loss_G + loss_MSE * args.param_MSE
                        is_first = False

                    if 'Laplacian' in archs[args.model][args.train_name]['Loss_CNN']:
                        edge_label = Laplacian_filter(train_label, edge_kernel, gray_k)
                        edge_g_output = Laplacian_filter(g_train_output, edge_kernel, gray_k)
                        loss_edge = criterion_L1(edge_label, edge_g_output)
                        loss_G = loss_edge * args.param_Laplacian if is_first else loss_G + loss_edge * args.param_Laplacian
                        is_first = False
                    
                    if 'DCT' in archs[args.model][args.train_name]['Loss_CNN']:
                        train_label_Y = rgb_to_ycbcr(train_label)
                        g_train_output_Y = rgb_to_ycbcr(g_train_output)
                        
                        train_label_dct_0 = torch.abs(dct_2d(train_label_Y[:, 0, :, :]).unsqueeze(1))
                        g_train_output_dct_0 = torch.abs(dct_2d(g_train_output_Y[:, 0, :, :]).unsqueeze(1))

                        if args.dct_binary:
                            train_label_dct = train_label_dct > args.dct_threshold
                            g_train_output_dct = g_train_output_dct > args.dct_threshold
                                        
                        train_label_dct = train_label_dct_0.type(torch.cuda.FloatTensor)
                        g_train_output_dct = g_train_output_dct_0.type(torch.cuda.FloatTensor)

                        if args.round_mask:
                            train_label_dct = train_label_dct * masking
                            g_train_output_dct = g_train_output_dct * masking

                        loss_DCT = criterion_L1(train_label_dct, g_train_output_dct)
                        loss_G = loss_DCT * args.param_DCT if is_first else loss_G + loss_DCT * args.param_DCT
                        is_first = False
                    
                    if 'Cross-Entropy' in archs[args.model][args.train_name]['Loss_CNN']:
                        print(train_label.shape, g_train_output.shape)
                        loss_Cross_Entropy = criterion_CE(train_label, g_train_output)
                        loss_G = loss_Cross_Entropy * args.param_Cross_Entropy if is_first else loss_G + loss_Cross_Entropy * args.param_Cross_Entropy
                        is_first = False
                    
                    loss_G.backward()
                    optimizer_G.step()

                else:
                    status = 'GAN'
                    g_train_output = Generator(train_data)

                    if i == 0:
                    
                        # ---------------------
                        #  Train Discriminator1
                        # ---------------------
                        if not archs[args.model][args.train_name]['Discriminator1'] == None:
                            optimizer_D_1.zero_grad()

                            dis_1_label_out = Variable(Discriminator_1(train_label), requires_grad=False)
                            dis_1_fake_out = Variable(Discriminator_1(g_train_output.detach()), requires_grad=True)

                            if not dis_1_label_out.shape == valid.shape:
                                continue

                            loss_D1_real = criterion_BCE(dis_1_label_out, valid)
                            loss_D1_fake = criterion_BCE(dis_1_fake_out, fake)

                            loss_D1 = torch.log((loss_D1_real + loss_D1_fake) / 2)

                            loss_D1.backward()
                            optimizer_D_1.step()

                        # ---------------------
                        #  Train Discriminator2
                        # ---------------------
                        if not archs[args.model][args.train_name]['Discriminator2'] == None:
                            optimizer_D_2.zero_grad()
                            train_label_Y = rgb_to_ycbcr(train_label)
                            g_train_output_Y = rgb_to_ycbcr(g_train_output.detach())
                            
                            train_label_dct_0 = torch.abs(dct_2d(train_label_Y[:, 0, :, :]).unsqueeze(1))
                            g_train_output_dct_0 = torch.abs(dct_2d(g_train_output_Y[:, 0, :, :]).unsqueeze(1))
                            
                            train_label_dct_1 = train_label_dct_0 > args.dct_threshold
                            g_train_output_dct_1 = g_train_output_dct_0 > args.dct_threshold
                            
                            train_label_dct_2 = train_label_dct_1.type(torch.cuda.FloatTensor)
                            g_train_output_dct_2 = g_train_output_dct_1.type(torch.cuda.FloatTensor)

                            train_label_dct = train_label_dct_2 * masking
                            g_train_output_dct = g_train_output_dct_2 * masking
                            
                            dis_2_label_out = Variable(Discriminator_2(train_label_dct), requires_grad=False)
                            dis_2_fake_out = Variable(Discriminator_2(g_train_output_dct), requires_grad=True)
                            
                            loss_D2_real = criterion_BCE(dis_2_label_out, valid)
                            loss_D2_fake = criterion_BCE(dis_2_fake_out, fake)
                            
                            loss_D2 = torch.log((loss_D2_real + loss_D2_fake) / 2)

                            loss_D2.backward()
                            optimizer_D_2.step()
                        
                    # ---------------------
                    #  Train Generator
                    # ---------------------
                    optimizer_G.zero_grad()

                    if 'VGG19' in archs[args.model][args.train_name]['Loss_GAN']:
                        gen_features = feature_extractor(g_train_output)
                        real_features = feature_extractor(train_label)
                        loss_VGG = criterion_MSE(gen_features, real_features.detach())
                        loss_mix = loss_VGG * args.param2_VGG if is_first else loss_mix + loss_VGG * args.param2_VGG
                        is_first = False

                    if 'L1' in archs[args.model][args.train_name]['Loss_GAN']:
                        loss_l1 = criterion_L1(train_label, g_train_output)
                        loss_mix = loss_l1 * args.param2_l1 if is_first else loss_mix + loss_l1 * args.param2_l1
                        is_first = False

                    if 'Laplacian' in archs[args.model][args.train_name]['Loss_GAN']:
                        edge_label = Laplacian_filter(train_label, edge_kernel, gray_k)
                        edge_g_output = Laplacian_filter(g_train_output, edge_kernel, gray_k)
                        loss_edge = criterion_L1(edge_label, edge_g_output)
                        loss_mix = loss_edge * args.param2_Laplacian if is_first else loss_mix + loss_edge * args.param2_Laplacian
                        is_first = False
                    
                    if 'DCT' in archs[args.model][args.train_name]['Loss_GAN']:
                        train_label_Y = rgb_to_ycbcr(train_label)
                        g_train_output_Y = rgb_to_ycbcr(g_train_output)
                        
                        train_label_dct_0 = torch.abs(dct_2d(train_label_Y[:, 0, :, :]).unsqueeze(1))
                        g_train_output_dct_0 = torch.abs(dct_2d(g_train_output_Y[:, 0, :, :]).unsqueeze(1))

                        if args.dct_binary:
                            train_label_dct = train_label_dct > args.dct_threshold
                            g_train_output_dct = g_train_output_dct > args.dct_threshold
                                        
                        train_label_dct = train_label_dct_0.type(torch.cuda.FloatTensor)
                        g_train_output_dct = g_train_output_dct_0.type(torch.cuda.FloatTensor)

                        if args.round_mask:
                            train_label_dct = train_label_dct * masking
                            g_train_output_dct = g_train_output_dct * masking

                        loss_DCT = criterion_L1(train_label_dct, g_train_output_dct)
                        loss_mix = loss_DCT * args.param2_DCT if is_first else loss_mix + loss_DCT * args.param2_DCT
                        is_first = False
                    
                    if i == 0:
                        if not archs[args.model][args.train_name]['Discriminator1'] == None:
                            dis_1_fake_out_mix = Variable(Discriminator_1(g_train_output), requires_grad=True) 
                            loss_D1_fake_mix = criterion_BCE(dis_1_fake_out_mix, valid)

                            loss_adv_D1 = torch.log(loss_D1_fake_mix)
                            loss_mix = - loss_adv_D1 * args.param2_D1 if is_first else loss_mix - loss_adv_D1 * args.param2_D1
                            is_first = False

                        if not archs[args.model][args.train_name]['Discriminator2'] == None:
                            g_train_output_Y_0 = rgb_to_ycbcr(g_train_output)
                            g_train_output_dct_00 = torch.abs(dct_2d(g_train_output_Y_0[:, 0, :, :]).unsqueeze(1))
                            g_train_output_dct_11 = g_train_output_dct_00 > args.dct_threshold
                            g_train_output_dct_22 = g_train_output_dct_11.type(torch.cuda.FloatTensor)
                            g_train_output_dct_33 = g_train_output_dct_22 * masking
                            dis_2_fake_out_mix_0 = Variable(Discriminator_2(g_train_output_dct_33), requires_grad=True) 
                            loss_D2_fake_mix = criterion_BCE(dis_2_fake_out_mix_0, valid)
                            
                            loss_adv_D2 = torch.log(loss_D2_fake_mix)
                            loss_mix = - loss_adv_D2 * args.param2_D2 if is_first else loss_mix - loss_adv_D2 * args.param2_D2
                            is_first = False

                    loss_mix.backward()
                    optimizer_G.step()
            
            # --------------
            #  Log Progress
            # --------------
            # --- Save Scalar ---
            if global_step % args.log_freq == 0:
                # Tensorboard
                if ((not args.step_mode) and epoch <= args.pretrain) or (args.step_mode and global_step <= args.pretrain_step): # CNN
                    writer.add_scalar("Loss_Pretrain/Generator", loss_G.item(), global_step)
                    if 'L1' in archs[args.model][args.train_name]['Loss_CNN']:
                        writer.add_scalar("Loss_Pretrain/L1", loss_l1.item(), global_step)
                    if 'MSE' in archs[args.model][args.train_name]['Loss_CNN']:
                        writer.add_scalar("Loss_Pretrain/MSE", loss_MSE.item(), global_step)
                    if 'Laplacian' in archs[args.model][args.train_name]['Loss_CNN']:
                        writer.add_scalar("Loss_Pretrain/Edge_Loss", loss_edge.item(), global_step)
                    if 'DCT' in archs[args.model][args.train_name]['Loss_CNN']:
                        writer.add_scalar("Loss_Pretrain/DCT_Loss", loss_DCT.item(), global_step)
                else:
                    writer.add_scalar("Loss_Generator/Generator", loss_mix.item(), global_step)
                    if 'L1' in archs[args.model][args.train_name]['Loss_GAN']:
                        writer.add_scalar("Loss_Generator/L1", loss_l1.item(), global_step)
                    if 'VGG19' in archs[args.model][args.train_name]['Loss_GAN']:
                        writer.add_scalar("Loss_Generator/VGG", loss_VGG.item(), global_step)
                    if not archs[args.model][args.train_name]['Discriminator1'] == None:
                        writer.add_scalar("Loss_Generator/Adv_D1", loss_adv_D1.item(), global_step)
                        writer.add_scalar("Loss_Discriminator1/Discriminator1", loss_D1.item(), global_step)
                    if not archs[args.model][args.train_name]['Discriminator2'] == None:
                        writer.add_scalar("Loss_Generator/Adv_D2", loss_adv_D2.item(), global_step)
                        writer.add_scalar("Loss_Discriminator2/Discriminator2", loss_D2.item(), global_step)

                # Logger
                if ((not args.step_mode) and epoch <= args.pretrain) or (args.step_mode and global_step <= args.pretrain_step): # CNN
                    txt3.append("{0:05d},{1:010d},{2:.5f},{3:.5f},{4:.5f}".format(epoch, global_step, loss_G.item(), 0, 0))
                else:
                    if archs[args.model][args.train_name]['Discriminator1'] == None and archs[args.model][args.train_name]['Discriminator2'] == None:
                        txt3.append("{0:05d},{1:010d},{2:.5f}".format(epoch, global_step, loss_mix.item()))
                    elif archs[args.model][args.train_name]['Discriminator1'] == None:
                        txt3.append("{0:05d},{1:010d},{2:.5f},{3:.5f}".format(epoch, global_step, loss_mix.item(), loss_D2.item()))
                    elif archs[args.model][args.train_name]['Discriminator2'] == None:
                        txt3.append("{0:05d},{1:010d},{2:.5f},{3:.5f}".format(epoch, global_step, loss_mix.item(), loss_D1.item()))
                    else:
                        txt3.append("{0:05d},{1:010d},{2:.5f},{3:.5f},{4:.5f}".format(epoch, global_step, loss_mix.item(), loss_D1.item(), loss_D2.item()))
                with codecs.open(os.path.join('.', args.output_name, train_name, 'logs{}.csv'.format(counter)), 'w', 'utf-8') as f:
                    f.write('\n'.join(txt3))
                
            # --- Save sample image ---
            if global_step % args.save_freq == 0 and global_step != start_step:
                save_image(g_train_output, os.path.join(args.output_name, train_name, str(global_step) + ".png"), normalize=False)
                if 'Kernel' in imgs:
                    kernel_image = imgs['Kernel'].to(device, dtype=torch.float)
                    save_image(kernel_image, os.path.join(args.output_name, train_name, str(global_step) + "_kernel.png"), normalize=False)
                
                # Tensorboard
                for i in range(min(args.batch_size, args.num_display)):
                    if not (args.kernel_mode == 'Estimate-Kernel-size' or args.kernel_mode == 'Estimate-Kernel-class'):
                        writer.add_image('Deblur/Image%d' %(i+1), g_train_output.detach().cpu().data.numpy()[i], global_step)
                        writer.add_image('Label/Image%d' %(i+1), train_label.cpu().numpy()[i], global_step)
                    if ((not args.step_mode) and epoch <= args.pretrain) or (args.step_mode and global_step <= args.pretrain_step):
                        if 'Laplacian' in archs[args.model][args.train_name]['Loss_CNN']:
                            writer.add_image('Deblur_Edge/Image%d' %(i+1), torch.clamp(edge_g_output, min=0.0, max=1.0).detach().cpu().numpy()[i], global_step)
                            writer.add_image('Label_Edge/Image%d' %(i+1), torch.clamp(edge_label, min=0.0, max=1.0).detach().cpu().numpy()[i], global_step)
                        if 'DCT' in archs[args.model][args.train_name]['Loss_CNN']:
                            writer.add_image('Deblur_DCT/Image%d' %(i+1), torch.clamp(g_train_output_dct, min=0.0, max=1.0).detach().cpu().numpy()[i], global_step)
                            writer.add_image('Label_DCT/Image%d' %(i+1), torch.clamp(train_label_dct, min=0.0, max=1.0).detach().cpu().numpy()[i], global_step)
                    else:
                        if not archs[args.model][args.train_name]['Discriminator2'] == None:
                            writer.add_image('Deblur_Y/Image%d' %(i+1), g_train_output_Y.detach().cpu().numpy()[i], global_step)
                            writer.add_image('Deblur_DCT0/Image%d' %(i+1), g_train_output_dct_0.detach().cpu().numpy()[i], global_step)
                            writer.add_image('Deblur_DCT1/Image%d' %(i+1), g_train_output_dct_1.detach().cpu().numpy()[i], global_step)
                            writer.add_image('Deblur_DCT/Image%d' %(i+1), g_train_output_dct.detach().cpu().numpy()[i], global_step)
                            writer.add_image('Label_Y/Image%d' %(i+1), train_label_Y.cpu().numpy()[i], global_step)
                            writer.add_image('Label_DCT0/Image%d' %(i+1), train_label_dct_0.cpu().numpy()[i], global_step)
                            writer.add_image('Label_DCT1/Image%d' %(i+1), train_label_dct_1.cpu().numpy()[i], global_step)
                            writer.add_image('Label_DCT/Image%d' %(i+1), train_label_dct.cpu().numpy()[i], global_step)
                         
            # --- Save model checkpoints ---
            if global_step % args.save_freq == 0 and global_step != start_step:
                torch.save(Generator.state_dict(), os.path.join(args.checkpoint, train_name, args.load_G + "_" + str(global_step) + ".pth"))
                if ((not args.step_mode) and epoch > args.pretrain) or (args.step_mode and global_step > args.pretrain_step):
                    if not archs[args.model][args.train_name]['Discriminator1'] == None:
                        torch.save(Discriminator_1.state_dict(), os.path.join(args.checkpoint, train_name, args.load_D1 + "_" + str(global_step) + ".pth"))
                    if not archs[args.model][args.train_name]['Discriminator2'] == None:
                        torch.save(Discriminator_2.state_dict(), os.path.join(args.checkpoint, train_name, args.load_D2 + "_" + str(global_step) + ".pth"))
                    
                with open(os.path.join(args.checkpoint, train_name, 'status.pickle'), 'wb') as f:
                    pickle.dump([counter, epoch, global_step] , f)
                
                txt2 = []
                txt2.append("# -- Status [%s] --" % version)
                txt2.append('|  |  |')
                txt2.append('| :--- | :--- |')
                txt2.append('| Mode | %s |' % status)
                txt2.append('| Start | %s |' % start_at)
                txt2.append('| End | Training ... |')
                txt2.append('| Epoch | %d/%d |' % (epoch, args.epoch))
                txt2.append('| Global Step | %d |' % global_step)
                with codecs.open(os.path.join('.', args.output_name, train_name, 'status{}.md'.format(counter)), 'w', 'utf-8') as f:
                    f.write('\n'.join(txt2))
                                
            global_step = global_step + 1
            
            if args.step_mode and args.step <= global_step:
                break
        
        
        if args.step_mode and args.step <= global_step:
            break

        if ((not args.step_mode) and epoch <= args.pretrain) or (args.step_mode and global_step <= args.pretrain_step):
            print(
                "[Epoch %d/%d] [Step: %d] [G loss: %f]"
                % (epoch, args.epoch, global_step, loss_G.item())
            )
        else:
            if archs[args.model][args.train_name]['Discriminator1'] != None and archs[args.model][args.train_name]['Discriminator2'] != None:
                print(
                    "[Epoch %d/%d] [Step: %d] [G loss: %f] [D1 loss: %f] [D2 loss: %f]"
                    % (epoch, args.epoch, global_step, loss_mix.item(), loss_D1.item(), loss_D2.item())
                )
            elif archs[args.model][args.train_name]['Discriminator1'] != None:
                print(
                    "[Epoch %d/%d] [Step: %d] [G loss: %f] [D loss: %f]"
                    % (epoch, args.epoch, global_step, loss_mix.item(), loss_D1.item())
                )
            else:
                print(
                    "[Epoch %d/%d] [Step: %d] [G loss: %f]"
                    % (epoch, args.epoch, global_step, loss_mix.item())
                )
            
    global_epoch = epoch
    
    torch.save(Generator.state_dict(), os.path.join(args.checkpoint, train_name, args.load_G + "_" + str(global_step) + ".pth"))
    if status == 'GAN':
        if not archs[args.model][args.train_name]['Discriminator1'] == None:
            torch.save(Discriminator_1.state_dict(), os.path.join(args.checkpoint, train_name, args.load_D1 + "_" + str(global_step) + ".pth"))
        if not archs[args.model][args.train_name]['Discriminator2'] == None:
            torch.save(Discriminator_2.state_dict(), os.path.join(args.checkpoint, train_name, args.load_D2 + "_" + str(global_step) + ".pth"))
    with open(os.path.join(args.checkpoint, train_name, 'status.pickle'), 'wb') as f:
        pickle.dump([counter, global_epoch, global_step] , f)
    
    end = datetime.datetime.now()
    end_at = end.strftime('%Y/%m/%d %H:%M:%S')
    
    txt2 = []
    txt2.append("# -- Status [%s] --" % version)
    txt2.append('|  |  |')
    txt2.append('| :--- | :--- |')
    txt2.append('| Mode | %s |' % status)
    txt2.append('| Start | %s |' % start_at)
    txt2.append('| End | %s |' % end_at)
    txt2.append('| Epoch | %d/%d |' % (epoch, args.epoch))
    txt2.append('| Global Step | %d |' % global_step)
    with codecs.open(os.path.join('.', args.output_name, train_name, 'status{}.md'.format(counter)), 'w', 'utf-8') as f:
        f.write('\n'.join(txt2))

    writer.close()

def test():
    print(' [*] Test Session')
    start_ALL = datetime.datetime.now()

    os.makedirs(args.result, exist_ok=True)
    os.makedirs(os.path.join(args.result, train_name), exist_ok=True)
    os.makedirs(os.path.join(args.data_folder, "time"), exist_ok=True)
    
    if torch.cuda.is_available(): 
        device = 'cuda'
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cudnn.benchmark=True
    else:
        device = 'cpu'

    print(' [*] Building model ...')
    if not archs[args.model][args.train_name]['Generator'] == None:
        Network = archs[args.model][args.train_name]['Generator']()
    else:
        print(' [!] Your Architecture has no Generator')
        exit()
    Network = Network.to(device)
    Network.eval()
    
    if not os.path.exists(os.path.join(args.checkpoint, train_name, 'status.pickle')):
        print(" [!] Load failed...", os.path.join(args.checkpoint, train_name, 'status.pickle'))
        exit()
    else:
        print(" [*] Load SUCCESS", train_name)
        with open(os.path.join(args.checkpoint, train_name, 'status.pickle'), 'rb') as f:
            counter, global_epoch, global_step = pickle.load(f)
        print(" [*] Global Epoch:", global_epoch, 'Step:', global_step)
        Network.load_state_dict(torch.load(os.path.join(args.checkpoint, train_name, args.load_G + "_" + str(global_step) + ".pth")), strict=False)
    
    dir_count = -1
    dir_path = args.test_dataset
    while '*' in dir_path:
        dir_path = os.path.dirname(dir_path)
        dir_count = dir_count + 1
    
    if dir_count == -1:
        print( ' [!] Please put test image into some folder')
        exit()

    print(" [*] Testing ...")

    testloader = DataLoader(
        TestDataset(args.test_dataset),
        batch_size=1,
        shuffle=False,
        num_workers=args.n_cpu
    )

    total_time = datetime.timedelta(microseconds=0)
    with torch.no_grad():
        for i, imgs in enumerate(testloader):
                    
            convert_image_size = False
            input_image = imgs['Input'].to(device)
            
            height = input_image.shape[2]
            width = input_image.shape[3]

            if input_image.shape[1] == 1:
                print(' [!] Size of Image might cause Error')
                continue

            if height % 4 != 0 or width % 4 != 0:
                print(' [!] Convert Image Size')
                convert_image_size = True

            if args.kernel_mode in ['Estimate-Kernel-size', 'Estimate-Kernel-class']:
                convert_image_size = False

            if args.patch:
                num_h = int(height // args.image_size)
                num_w = int(width // args.image_size)

                if args.kernel_mode in ['Estimate-Kernel-size', 'Estimate-Kernel-class']:
                    outputs = torch.zeros(num_h * num_w)
                    
                    start = datetime.datetime.now()
                    for h in range(num_h):
                        for w in range(num_w):
                            outputs[h*num_w + w] = Network(input_image[:, :, h*args.image_size:(h+1)*args.image_size, w*args.image_size:(w+1)*args.image_size])
                    end = datetime.datetime.now()

            else:
                if convert_image_size:
                    input_image = F.interpolate(input_image, (height - (height % 4), width - (width % 4)), mode='nearest')
                
                start = datetime.datetime.now()
                for scale in range(1 - args.scale, 1):
                    if args.scale != 1:
                        data_scale1 = F.interpolate(input_image, (int(height*(2**scale)), int(width*(2**scale))), mode='nearest')

                        if scale == 1 - args.scale:
                            data_scale2 = data_scale1.clone()
                        else:
                            data_scale2 = F.interpolate(outputs.detach(), (int(height*(2**scale)), int(width*(2**scale))), mode='nearest')
                        
                        input_image_scale = torch.cat((data_scale1, data_scale2), dim=1)
                    else:
                        input_image_scale = input_image

                    outputs = Network(input_image_scale)

                end = datetime.datetime.now()

                if convert_image_size:
                    outputs = F.interpolate(outputs, (height, width), mode='nearest')
                    
            dir_path = imgs['Filename'][0]
            for j in range(dir_count):
                dir_path = os.path.dirname(dir_path)
            
            if dir_count == 0:
                save_dir = os.path.join(args.result, train_name)
            else:
                os.makedirs(os.path.join(args.result, train_name, os.path.basename(dir_path)), exist_ok=True)
                save_dir = os.path.join(args.result, train_name, os.path.basename(dir_path))
            
            if args.kernel_mode in ['Estimate-Kernel-size', 'Estimate-Kernel-class']:
                if args.patch:
                    out_kernel_size = outputs.cpu().numpy()

                    np.save(os.path.join(save_dir, os.path.splitext(os.path.basename(imgs['Filename'][0]))[0]), out_kernel_size)
                    print('[{0:05d}] {1}\n Kernel Size Head: {2}\n Time: {3}'.format(i+1, imgs['Filename'][0], out_kernel_size[:5], end - start))
                else:
                    out_kernel_size = outputs[0].cpu().numpy()

                    if os.path.exists(os.path.join(save_dir, 'kernel_size.npy')):
                        out_sizes = np.load(os.path.join(save_dir, 'kernel_size.npy'))
                        out_kernel_sizes = np.append(out_sizes, out_kernel_size)
                    else:
                        out_kernel_sizes = out_kernel_size
                    np.save(os.path.join(save_dir, 'kernel_size'), out_kernel_sizes)
                    print('[{0:05d}] {1} Kernel Size: {2} Time: {3}'.format(i+1, imgs['Filename'][0], out_kernel_size, end - start))
            else:                                
                out_image = outputs[0].cpu().numpy().transpose(1, 2, 0)

                cv2.imwrite(os.path.join(save_dir, os.path.basename(imgs['Filename'][0])), cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB) * 255)
                print('[{0:05d}] {1} {2} {3}'.format(i+1, imgs['Filename'][0], input_image.shape, end - start))
            
            total_time = total_time + end - start
    
    end_ALL = datetime.datetime.now()

    if os.path.exists(os.path.join(args.data_folder, 'time', train_name + '.pickle')):
        with open(os.path.join(args.data_folder, 'time', train_name + '.pickle'), 'rb') as f:
            times = pickle.load(f)
    else:
        times = {'dataset': [], 'total_time': [], 'time_all': [], 'number': [], 'average_time': datetime.timedelta(microseconds=0)}
    
    times['dataset'].append(save_dir)
    times['total_time'].append(total_time)
    times['time_all'].append(end_ALL - start_ALL)
    times['number'].append(len(testloader))
    times['average_time'] = total_time / len(testloader)

    with open(os.path.join(args.data_folder, 'time', train_name + '.pickle'), 'wb') as f:
        pickle.dump(times, f)
    
    print('Total Time:', total_time, 'Average Time:', total_time / len(testloader))
    print('Total Time:', end_ALL - start_ALL, 'Average Time:', (end_ALL - start_ALL) / len(testloader))

def save_parameter(args, archs, counter):
    if os.path.exists(os.path.join('.', args.output_name, train_name, 'parameter.pickle')):
        with open(os.path.join('.', args.output_name, train_name, 'parameter.pickle'), 'rb') as f:
            parameter = pickle.load(f)
    else:
        parameter = []

    parameter.append(args.__dict__)
    
    with open(os.path.join('.', args.output_name, train_name, 'parameter.pickle'), 'wb') as f:
        pickle.dump(parameter , f)
    
    with open(os.path.join('.', args.output_name, train_name, 'parameter{}.json'.format(counter)), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    txt = []
    param_sum, param2_sum = 0, 0
    param_MSE = args.param_MSE if 'MSE' in archs[args.model][args.train_name]['Loss_CNN'] else 0
    param_l1 = args.param_l1 if 'L1' in archs[args.model][args.train_name]['Loss_CNN'] else 0
    param_VGG = args.param_VGG if 'VGG19' in archs[args.model][args.train_name]['Loss_CNN'] else 0
    param_Laplacian = args.param_Laplacian if 'Laplacian' in archs[args.model][args.train_name]['Loss_CNN'] else 0
    param_DCT = args.param_DCT if 'DCT' in archs[args.model][args.train_name]['Loss_CNN'] else 0
    param_Cross_Entropy = args.param_Cross_Entropy if 'Cross-Entropy' in archs[args.model][args.train_name]['Loss_CNN'] else 0
    param2_MSE = args.param2_MSE if 'MSE' in archs[args.model][args.train_name]['Loss_GAN'] else 0
    param2_l1 = args.param2_l1 if 'L1' in archs[args.model][args.train_name]['Loss_GAN'] else 0
    param2_VGG = args.param2_VGG if 'VGG19' in archs[args.model][args.train_name]['Loss_GAN'] else 0
    param2_Laplacian = args.param2_Laplacian if 'Laplacian' in archs[args.model][args.train_name]['Loss_GAN'] else 0
    param2_DCT = args.param2_DCT if 'DCT' in archs[args.model][args.train_name]['Loss_GAN'] else 0
    param2_D1 = args.param2_D1 if not archs[args.model][args.train_name]['Discriminator1'] == None else 0
    param2_D2 = args.param2_D2 if not archs[args.model][args.train_name]['Discriminator2'] == None else 0
    param_sum = (param_MSE + param_l1 + param_VGG + param_Laplacian + param_DCT) / 100
    param2_sum = (param2_MSE + param2_l1 + param2_VGG + param2_Laplacian + param2_DCT + param2_D1 + param2_D2) / 100
    if param_sum == 0:
        param_sum = 1
    if param2_sum == 0:
        param2_sum = 1

    txt.append("# --- Pytorch Controller [%s] ---" % version)
    txt.append("# " + train_name)
    txt.append("\n## -- Basic Setting --")
    txt.append('|  |  |')
    txt.append('| :--- | :--- |')
    txt.append("| gpu_id | {} |".format(args.gpu_id))
    txt.append("| train_name | {} |".format(train_name))
    txt.append("| output_name| {} |".format(args.output_name))
    txt.append("| log_freq | {} |".format(args.log_freq))
    txt.append("| save_freq | {} |".format(args.save_freq))
    txt.append("| checkpoint | {} |".format(args.checkpoint))
    txt.append("| load_G | {} |".format(args.load_G))
    txt.append("| load_D1 | {} |".format(args.load_D1))
    txt.append("| load_D2 | {} |".format(args.load_D2))
    txt.append("| num_display | {} |".format(args.num_display))
    txt.append("\n## -- Kernel Generation --")
    txt.append('|  |  |')
    txt.append('| :--- | ---: |')
    txt.append("| kernel_mode | {} |".format(args.kernel_mode))
    txt.append("| linear_rate | {} |".format(args.linear_rate))
    txt.append("| kernel_flame | {} |".format(args.kernel_flame))
    txt.append("| kernel_max | {} |".format(args.kernel_max))
    txt.append("| kernel_min | {} |".format(args.kernel_min))
    txt.append("| all_sigma | {} |".format(args.all_sigma))
    txt.append("| sigma_min | {} |".format(args.sigma_min))
    txt.append("| sigma_max | {} |".format(args.sigma_max))
    txt.append("| add_angle_min | {} |".format(args.add_angle_min))
    txt.append("| add_angle_max | {} |".format(args.add_angle_max))
    txt.append("| cache | {} |".format(args.cache))
    txt.append("\n## -- Hyper Parameter --")
    txt.append('|  |  |')
    txt.append('| :--- | ---: |')
    txt.append("| epoch | {} |".format(args.epoch))
    txt.append("| pretrain | {} |".format(args.pretrain))
    txt.append("| step_mode | {} |".format(args.step_mode))
    txt.append("| step | {} |".format(args.step))
    txt.append("| pretrain_step | {} |".format(args.pretrain_step))
    txt.append("| learning_rate | {} |".format(args.learning_rate))
    txt.append("| b1 | {} |".format(args.b1))
    txt.append("| b2 | {} |".format(args.b2))
    txt.append("| batch_size | {} |".format(args.batch_size))
    txt.append("### - Pretrain -")
    txt.append('|  |  |  |')
    txt.append('| :--- | ---: | ---: |')
    txt.append("| param_MSE | {0:02.3f} | {1:.3f} % |".format(param_MSE, param_MSE / param_sum))
    txt.append("| param_l1 | {0:02.3f} | {1:.3f} % |".format(param_l1, param_l1 / param_sum))
    txt.append("| param_VGG | {0:02.3f} | {1:.3f} % |".format(param_VGG, param_VGG / param_sum))
    txt.append("| param_Laplacian | {0:02.3f} | {1:.3f} % |".format(param_Laplacian, param_Laplacian / param_sum))
    txt.append("| param_DCT | {0:02.3f} | {1:.3f} % |".format(param_DCT, param_DCT / param_sum))
    txt.append("| param_Cross_Entropy | {0:02.3f} | {1:.3f} % |".format(param_Cross_Entropy, param_Cross_Entropy / param_sum))
    txt.append("### - GAN -")
    txt.append('|  |  |  |')
    txt.append('| :--- | ---: | ---: |')
    txt.append("| param_MSE | {0:02.3f} | {1:.3f} % |".format(param2_MSE, param2_MSE / param2_sum))
    txt.append("| param_l1 | {0:02.3f} | {1:.3f} % |".format(param2_l1, param2_l1 / param2_sum))
    txt.append("| param_VGG | {0:02.3f} | {1:.3f} % |".format(param2_VGG, param2_VGG / param2_sum))
    txt.append("| param_Laplacian | {0:02.3f} | {1:.3f} % |".format(param2_Laplacian, param2_Laplacian / param2_sum))
    txt.append("| param_DCT | {0:02.3f} | {1:.3f} % |".format(param2_DCT, param2_DCT / param2_sum))
    txt.append("| param_D1 | {0:02.3f} | {1:.3f} % |".format(param2_D1, param2_D1 / param2_sum))
    txt.append("| param_D2 | {0:02.3f} | {1:.3f} % |".format(param2_D2, param2_D2 / param2_sum))
    txt.append("| dct_binary | {} |".format(args.dct_binary))
    txt.append("| param_dct_threshold | {} |".format(args.dct_threshold))
    txt.append("| round_mask | {} |".format(args.round_mask))
    txt.append("| param_mask_threshold | {} |".format(args.mask_threshold))
    txt.append("\n## -- Dataset --")
    txt.append('|  |  |')
    txt.append('| :--- | :--- |')
    txt.append("| dataset | {} |".format(args.dataset))
    txt.append("| dataset_train | {} |".format(args.dataset_train))
    txt.append("| image_size | {} |".format(args.image_size))
    txt.append("| n_cpu | {} |".format(args.n_cpu))
    txt.append("\n## -- Test --")
    txt.append('|  |  |')
    txt.append('| :--- | :--- |')
    txt.append("| test | {} |".format(args.test))
    txt.append("| test_dataset | {} |".format(args.test_dataset))
    txt.append("| result | {} |".format(args.result))

    with codecs.open(os.path.join('.', args.output_name, train_name, 'parameter{}.md'.format(counter)), 'w', 'utf-8') as f:
        f.write('\n'.join(txt))

if __name__ == '__main__':
    if args.test:
        test()
    else:
        os.makedirs(os.path.join('.', args.output_name), exist_ok=True)
        os.makedirs(os.path.join('.', args.output_name, train_name), exist_ok=True)

        train()
