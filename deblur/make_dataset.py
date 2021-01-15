import glob
import os
import tqdm
import argparse
import cv2
import numpy as np
import random
import pickle
import json
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import norm

from kernel_generation import *

parser = argparse.ArgumentParser()
parser.add_argument("--mode",type=str,default="Adjust")
parser.add_argument("--read_path",type=str,default="/home/tomosada/MSCOCO/train2017/*.jpg")
parser.add_argument("--save_path",type=str,default="/home/tomosada/MSCOCO/train_apply2")

# Adjust
parser.add_argument("--min_size",type=int,default=384)

# Blur-Generation
parser.add_argument('--kernel_mode',type=str,default="No-Kernel")
parser.add_argument('--linear_rate',default=0,type=float)
parser.add_argument("--kernel_flame",default=61,type=int)
parser.add_argument("--kernel_max",default=51,type=int)
parser.add_argument("--kernel_min",default=11,type=int)
parser.add_argument("--all_sigma",default=1,type=int)
parser.add_argument("--sigma_min",default=1,type=int)
parser.add_argument("--sigma_max",default=20,type=int)
parser.add_argument("--add_angle_min",default=0,type=int)
parser.add_argument("--add_angle_max",default=0,type=int)

# Blur-Classification
parser.add_argument('--patch', action='store_true')

# Blur-Classification-All
parser.add_argument("--target_dataset_index",default=0,type=int)
parser.add_argument('--theory', action='store_true')
parser.add_argument('--theory_mode',type=str,default="Gaussian")
parser.add_argument("--min",default=0,type=int)
parser.add_argument("--max",default=30,type=int)
parser.add_argument("--average",default=9,type=float)
parser.add_argument("--sigma",default=3,type=float)
parser.add_argument("--th_min",default=3,type=int)
parser.add_argument("--th_max",default=7,type=int)

# Dataset
parser.add_argument("--image_size",default=256,type=int)
args = parser.parse_args()

version = 'Version 2.0'
print(" [*] Dataset Generator [%s] powered by Tomosada" % version)

os.makedirs(args.save_path, exist_ok=True)

dataset_paths = [
    "/home/tomosada/GOPRO_kernel/Kernel-Prediction-Normal-3/*/*.npy",
    "/home/tomosada/DVD_kernel/Kernel-Prediction-Normal-3/*/*.npy",
    "/home/tomosada/NFS_kernel/Kernel-Prediction-Normal-3/*/*.npy",
    "/home/tomosada/HIDE_kernel/Kernel-Prediction-Normal-3/*.npy",
]

class Theory:
    def __init__(self, args):
        self.average = args.average
        self.sigma = args.sigma
        self.th_max = args.th_max
        self.th_min = args.th_min

    def Gaussian(self, min_size, max_size):
        number = np.arange(min_size, max_size + 1, 1)
        size_path = norm.pdf(number, self.average, self.sigma)
        return size_path
    
    def Step(self, min_size, max_size):
        size_path = np.zeros(max_size - min_size + 1)
        print(size_path.shape)
        size_path[self.th_min - min_size:self.th_max - min_size] = np.ones(self.th_max - self.th_min)
        return size_path

def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            size = pickle.load(f)
        return size
    else:
        return []

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    paths = sorted(glob.glob(os.path.join(args.read_path)))

    save_dir = args.save_path

    if args.mode == 'Adjust':
        count = 0
        for i, path in enumerate(paths):
            img = Image.open(path)
            if min(img.size) >= args.min_size and img.mode == 'RGB':
                print('[{0:010d}] {1}'.format(i+1, path))
                img.save(os.path.join(args.save_path, os.path.basename(path)), quality=95)
            else:
                count = count + 1
                print('[{0:010d}] {1} No Good'.format(count, path))
        
            print(' [*] {0}percent of dataset is available'.format(100 - 100*count/len(paths)))
    
    elif args.mode == 'Blur-Generation':
        kernel_sizes = np.zeros(len(paths))
        for i, path in enumerate(paths):
            print('[{0:010d}] {1}'.format(i+1, path))
            img = Image.open(path)
            img_np = np.asarray(img)

            width, height = img.size
            origin_y = random.randint(0, height - args.image_size - 1)
            origin_x = random.randint(0, width - args.image_size - 1)
            img_np = img_np[origin_y : origin_y + args.image_size, origin_x : origin_x + args.image_size, :]
            
            if random.random() < args.linear_rate:
                kernel, kernel_img, kernel_size = get_random_linear_kernel(args)
            else:
                kernel, kernel_img, kernel_size = get_random_non_linear_kernel(args)
            
            kernel_sizes[i] = kernel_size

            blur = cv2.filter2D(img_np, -1, kernel)
            cv2.imwrite(os.path.join(args.save_path, os.path.basename(path)), blur)
            filename, ext = os.path.splitext(os.path.basename(path))
            cv2.imwrite(os.path.join(args.save_path, filename + '_kernel' + ext), kernel_img*255)
        
        np.save(os.path.join(args.save_path, 'kernel_size'), kernel_sizes)
    
    elif args.mode in ['Blur-Classification', 'Blur-Classification-All']:
        if args.mode == 'Blur-Classification':
            dir_count = 0
            dir_path = args.read_path
            while '*' in dir_path:
                dir_path = os.path.dirname(dir_path)
                print(dir_path)
                dir_count = dir_count + 1
                
            if dir_count == 0:
                print( ' [!] Please put data into some folder')
                exit()
            
            if dir_count == 0:
                save_dir = args.save_path
            else:
                os.makedirs(os.path.join(args.save_path, os.path.basename(dir_path)), exist_ok=True)
                save_dir = os.path.join(args.save_path, os.path.basename(dir_path))

            dataset_name = os.path.basename(os.path.dirname(dir_path))

        elif args.mode == 'Blur-Classification-All':
            paths = []
            index = 0
            index_start = np.zeros(len(dataset_paths))
            index_end = np.zeros(len(dataset_paths))

            for i, dataset_path in enumerate(dataset_paths):
                path = sorted(glob.glob(dataset_path))

                index_start[i] = index
                index_end[i] = index + len(path) - 1
                index += len(path)

                paths.extend(path)
            
            dataset_name = 'All'
            save_dir = args.save_path

        if not args.patch:
            dataset_name = dataset_name + '_img'
        
        count = 0
        for i, path in enumerate(tqdm.tqdm(paths)):
            imply = np.load(path)

            if not args.patch:
                imply = np.average(imply)

                size_paths = load_pickle(os.path.join(save_dir, "Size" +  str(int(np.round(imply))) + ".pickle"))
                size_path = load_pickle(os.path.join(save_dir, "Size" +  str(int(np.round(imply))) + "_Dataset" + str(count) + ".pickle"))
            
                size_paths.append(path)
                size_path.append(path)

                save_pickle(os.path.join(save_dir, "Size" +  str(int(np.round(imply))) + ".pickle"), size_paths)
                save_pickle(os.path.join(save_dir, "Size" +  str(int(np.round(imply))) + "_Dataset" + str(count) + ".pickle"), size_path)
                
            if i == index_end[count]:
                count += 1

            if i == 0:
                implys = imply
            else:
                implys = np.append(implys, imply)
            
        print('All avg Size: {0} All std Size: {1}'.format(np.average(implys), np.std(implys)))

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.hist(implys, bins=50)
        ax.set_title(dataset_name + ' Dataset')
        ax.set_xlim(-5, 35)
        ax.set_xlabel('Kernel Size')
        ax.set_ylabel('Number')

        fig.savefig(os.path.join(save_dir, dataset_name + ".png"))

        min_size = int(np.round(np.min(implys)))
        max_size = int(np.round(np.max(implys)))

        min_size = max(min_size, args.min)
        max_size = min(max_size, args.max)

        with open(os.path.join(save_dir, "output.pickle"), 'wb') as f:
            pickle.dump([min_size, max_size], f)

        rates = np.zeros(max_size - min_size + 1)
        target_hist = np.zeros(max_size - min_size + 1)
        if args.mode == 'Blur-Classification-All':
            if args.theory:
                method = Theory(args)
                if args.theory_mode == 'Gaussian':
                    size_path = method.Gaussian(min_size, max_size)
                elif args.theory_mode == 'Step':
                    size_path = method.Step(min_size, max_size)
                
            for i, size in enumerate(range(min_size, max_size + 1)):

                size_paths = load_pickle(os.path.join(save_dir, "Size" +  str(size) + ".pickle"))
                
                if args.theory:
                    rates[i] = len(size_paths) / size_path[i]
                    target_hist[i] = size_path[i]
                    print('Kernel Size: {0} All Dataset: {1} Target Dataset: {2}'.format(size, len(size_paths), size_path[i]))
                
                else:
                    size_path = load_pickle(os.path.join(save_dir, "Size" +  str(size) + "_Dataset" + str(args.target_dataset_index) + ".pickle"))
                    rates[i] = len(size_paths) / len(size_path)
                    target_hist[i] = len(size_path)
                    print('Kernel Size: {0} All Dataset: {1} Target Dataset: {2}'.format(size, len(size_paths), len(size_path)))
            min_rate = min(rates[rates != 0])

            count = 0
            for i, size in enumerate(range(min_size, max_size + 1)):
                size_path = load_pickle(os.path.join(save_dir, "Size" +  str(size) + ".pickle"))
                
                random_path = random.sample(size_path, int(target_hist[i] * min_rate))

                count += int(target_hist[i] * min_rate)

                save_pickle(os.path.join(save_dir, "output_" + str(size) + ".pickle"), random_path)
            
            dataset_name = 'dataset_hist'
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(np.arange(min_size, max_size + 1, 1), target_hist*min_rate)
            ax.set_title(dataset_name + ' Dataset')
            ax.set_xlim(-5, 35)
            ax.set_xlabel('Kernel Size')
            ax.set_ylabel('Number')
            fig.savefig(os.path.join(save_dir, dataset_name + ".png"))

    else:
        print(' [!] Nothing to do')
    
    with open(os.path.join(save_dir, 'parameter.pickle'), 'wb') as f:
        pickle.dump(args.__dict__ , f)
    
    with open(os.path.join(save_dir, 'parameter.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
