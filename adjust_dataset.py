import os
import glob
import cv2

basepath = '/home/tomosada/real_dataset/*.jpg'
savepath = '/home/tomosada/real_dataset_4'

paths = sorted(glob.glob(os.path.join(basepath)))

os.makedirs(os.path.join(savepath), exist_ok=True)

for i, path in enumerate(paths):
    print(' [{0:05d}] {1}'.format(i, path))
    img = cv2.imread(path)

    if img.shape[0] % 4 != 0 or img.shape[1] % 4 != 0:
        print(' [!] This size of image might cause error')
        continue
    
    if len(img.shape) == 1:
        print(' [!] This image is not color')
        continue
    
    cv2.imwrite(os.path.join(savepath, os.path.basename(path)), img)
    
