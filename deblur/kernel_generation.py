import numpy as np
import cv2
import math
import random
import tqdm

def make_linear_kernel(kernel_size, angle, scale):
    kernel_base = np.zeros([kernel_size, kernel_size])
    kernel_base[:,kernel_size // 2] = np.ones(kernel_size)

    kernel_base = kernel_base / kernel_size
    height, width = kernel_base.shape

    center = (int(width/2), int(height/2))

    trans = cv2.getRotationMatrix2D(center, angle , scale)
    kernel = cv2.warpAffine(kernel_base, trans, (width,height))

    return kernel

def get_all_linear_kernel(args):
    scale = 1.0
    kernels = np.zeros([int((args.kernel_max - args.kernel_min)/2 + 1) * 180, args.kernel_flame, args.kernel_flame, 1])
    kernel_sizes = np.zeros(int((args.kernel_max - args.kernel_min)/2 + 1) * 180)

    for kernel_size in tqdm.tqdm(range(args.kernel_min, args.kernel_max + 1)):
        if kernel_size % 2 == 0:
            continue
        
        for angle in range(180):
            kernel = make_linear_kernel(kernel_size, angle, scale)
            pad_kernel = np.pad(kernel, (int((args.kernel_flame - kernel_size) / 2), int((args.kernel_flame - kernel_size) / 2)), 'constant')
            pad_kernel = np.expand_dims(pad_kernel, axis=-1)

            kernels[int((kernel_size - args.kernel_min)/2) * 180 + angle, :, :, :] = pad_kernel
            kernel_sizes[int((kernel_size - args.kernel_min)/2) * 180 + angle] = kernel_size
    
    return kernels, kernel_sizes

def get_random_linear_kernel(args):
    scale = 1.0
    kernel_size = random.randint(args.kernel_min, args.kernel_max)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    angle = random.randint(0, 179)
    kernel = make_linear_kernel(kernel_size, angle, scale)
    pad_kernel = np.pad(kernel, (int((args.kernel_flame - kernel_size) / 2), int((args.kernel_flame - kernel_size) / 2)), 'constant')
    return pad_kernel, pad_kernel*kernel_size, kernel_size

def gaussian_func(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

def make_nonlinear_kernel(args, kernel_size, sigma_min, sigma_max, add_angle):
    kernel_max = args.kernel_max
    kernel_flame = args.kernel_flame
    
    frame_size = kernel_flame * 4

    kernel_base = np.zeros([frame_size, frame_size])

    x = int(frame_size / 2)
    y = int(frame_size / 2)

    angle_init = random.randint(-180, 180)
    sigma = random.uniform(sigma_min, sigma_max)

    probability_angle = np.random.rand(360)
    probability_index = np.arange(-180, 180)
    pdf_angle = gaussian_func(probability_index, angle_init, sigma)
    
    for i in range(kernel_size):
        probability = probability_angle * pdf_angle
        angle_index = np.argmax(probability)
        angle = probability_index[angle_index] + add_angle
        if angle >= 0:
            angle = angle % 360
        elif angle < 0:
            angle = angle % 360 - 360
        
        if (0 <= angle and angle < 90) or (-360 <= angle and angle < -270):
            x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = int(x) + 1, int(y) + 1, int(x) + 1, int(y), int(x), int(y) + 1, int(x), int(y)
            S1 = math.fabs(x - int(x) + math.cos(math.radians(angle)) - 0.5) * math.fabs(y - int(y) + math.sin(math.radians(angle)) - 0.5) # (x+1, y+1)
            S2 = math.fabs(x - int(x) + math.cos(math.radians(angle)) - 0.5) * math.fabs(int(y) - y + math.sin(math.radians(angle)) + 1.5) # (x, y+-1)
            S3 = math.fabs(int(x) - x + math.cos(math.radians(angle)) + 1.5) * math.fabs(y - int(y) + math.sin(math.radians(angle)) - 0.5) # (x+-1, y)
            S4 = math.fabs(int(x) - x + math.cos(math.radians(angle)) + 1.5) * math.fabs(int(y) - y + math.sin(math.radians(angle)) + 1.5) # (x, y)
        elif (-90 <= angle and angle < 0) or (270 <= angle and angle < 360):
            x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = int(x) + 1, int(y) - 1, int(x) + 1, int(y), int(x), int(y) - 1, int(x), int(y)
            S1 = math.fabs(x - int(x) + math.cos(math.radians(angle)) - 0.5) * math.fabs(int(y) - y + math.sin(math.radians(angle)) + 0.5) # (x+1, y+1)
            S2 = math.fabs(x - int(x) + math.cos(math.radians(angle)) - 0.5) * math.fabs(y - int(y) + math.sin(math.radians(angle)) + 0.5) # (x, y+-1)
            S3 = math.fabs(int(x) - x + math.cos(math.radians(angle)) + 1.5) * math.fabs(int(y) - y - math.sin(math.radians(angle)) + 0.5) # (x+-1, y)        
            S4 = math.fabs(int(x) - x + math.cos(math.radians(angle)) + 1.5) * math.fabs(y - int(y) + math.sin(math.radians(angle)) + 0.5) # (x, y)
        elif (90 <= angle and angle < 180) or (-270 <= angle and angle < -180):
            x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = int(x) - 1, int(y) + 1, int(x) - 1, int(y), int(x), int(y) + 1, int(x), int(y)
            S1 = math.fabs(int(x) - x - math.cos(math.radians(angle)) + 0.5) * math.fabs(y - int(y) + math.sin(math.radians(angle)) - 0.5) # (x-1, y+1)
            S2 = math.fabs(int(x) - x - math.cos(math.radians(angle)) + 0.5) * math.fabs(int(y) - y + math.sin(math.radians(angle)) + 1.5) # (x-1, y)
            S3 = math.fabs(x - int(x) + math.cos(math.radians(angle)) + 0.5) * math.fabs(y - int(y) + math.sin(math.radians(angle)) - 0.5) # (x, y+1)
            S4 = math.fabs(x - int(x) + math.cos(math.radians(angle)) + 0.5) * math.fabs(int(y) - y + math.sin(math.radians(angle)) + 1.5) # (x, y)
        elif (-180 <= angle and angle < -90) or (180 <= angle and angle < 270):
            x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = int(x) - 1, int(y) - 1, int(x) - 1, int(y), int(x), int(y) - 1, int(x), int(y)
            S1 = math.fabs(int(x) - x - math.cos(math.radians(angle)) + 0.5) * math.fabs(int(y) - y + math.sin(math.radians(angle)) + 0.5) # (x-1, y-1)
            S2 = math.fabs(int(x) - x - math.cos(math.radians(angle)) + 0.5) * math.fabs(y - int(y) + math.sin(math.radians(angle)) + 0.5) # (x-1, y)
            S3 = math.fabs(x - int(x) + math.cos(math.radians(angle)) + 0.5) * math.fabs(int(y) - y - math.sin(math.radians(angle)) + 0.5) # (x, y-1)
            S4 = math.fabs(x - int(x) + math.cos(math.radians(angle)) + 0.5) * math.fabs(y - int(y) + math.sin(math.radians(angle)) + 0.5) # (x, y)

        S_sum = S1 + S2 + S3 + S4

        kernel_base[x_1, y_1] = kernel_base[x_1, y_1] + (S1 / S_sum)
        kernel_base[x_2, y_2] = kernel_base[x_2, y_2] + (S2 / S_sum)
        kernel_base[x_3, y_3] = kernel_base[x_3, y_3] + (S3 / S_sum)
        kernel_base[x_4, y_4] = kernel_base[x_4, y_4] + (S4 / S_sum)

        x = x + math.cos(math.radians(angle))
        y = y + math.sin(math.radians(angle))

        probability_angle = np.random.rand(360)
        sigma = random.uniform(sigma_min, sigma_max)
        pdf_angle = gaussian_func(probability_index, angle, sigma)
    
    kernel_before = kernel_base / kernel_size
    
    kernel_image_before = kernel_base
    mu = cv2.moments(kernel_image_before, False)
    center_y, center_x = int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])
    
    kernel = kernel_before[center_x - int(kernel_flame/2):center_x + int(kernel_flame/2) + 1, center_y - int(kernel_flame/2):center_y + int(kernel_flame/2) + 1]
    kernel_image = kernel_image_before[center_x - int(kernel_flame/2):center_x + int(kernel_flame/2) + 1, center_y - int(kernel_flame/2):center_y + int(kernel_flame/2) + 1]
    return kernel, kernel_image

def get_cache_non_linear_kernel(args):
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    add_angle_min = args.add_angle_min
    add_angle_max = args.add_angle_max

    kernels = np.zeros([args.cache, args.kernel_flame, args.kernel_flame, 1])
    kernel_sizes = np.zeros(args.cache)

    span = int(args.cache / (args.kernel_max - args.kernel_min))
    kernel_size = args.kernel_min
    for i in tqdm.tqdm(range(args.cache)):
        # 26 - 51
        add_angle = random.randint(add_angle_min, add_angle_max)
        if i % span == 0:
            kernel_size = kernel_size + 1
        if args.all_sigma == 1:
            sigma_max = random.randint(1, 360)
            sigma_min = random.randint(1, sigma_max)
        tmp_kernel, _ = make_nonlinear_kernel(args, kernel_size, sigma_min, sigma_max, add_angle)
        
        kernels[i, :, :, 0] = tmp_kernel
        kernel_sizes[i] = kernel_size    
    
    return kernels, kernel_sizes

def get_random_non_linear_kernel(args):
    scale = 1.0
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    add_angle_min = args.add_angle_min
    add_angle_max = args.add_angle_max
    add_angle = random.randint(add_angle_min, add_angle_max)
    kernel_size = random.randint(args.kernel_min, args.kernel_max)
    
    if args.all_sigma == 1:
        sigma_max = random.randint(1, 360)
        sigma_min = random.randint(1, sigma_max)
    
    kernel, kernel_image = make_nonlinear_kernel(args, kernel_size, sigma_min, sigma_max, add_angle)
    return kernel, kernel_image, kernel_size