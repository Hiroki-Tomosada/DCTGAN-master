# GAN-Based Image Deblurring Using DCT Discriminator
## Installation
Clone this project to your machine. 

```bash
git clone https://github.com/Hiroki-Tomosada/DCTGAN-master.git
cd DCTGAN-master/deblur/
```

## Dataset
The datasets for training can be downloaded via the links below:

- [GOPRO](https://seungjunnah.github.io/Datasets/gopro)
- [DVD](https://drive.google.com/file/d/1bpj9pCcZR_6-AHb5aNnev5lILQbH8GMZ/view)
- [NFS](https://drive.google.com/file/d/1Ut7qbQOrsTZCUJA_mJLptRMipD8sJzjy/view)
- [HIDE](https://www.dropbox.com/s/04w3wqxcuin9dy8/HIDE_dataset.zip?dl=0)

## Train
```bash
python main-Pytorch.py --data_folder <MODEL_PATH> --dataset_train <DATASET_PATH>
```

## Test
### GOPRO
```bash
python main-Pytorch.py --data_folder <MODEL_PATH> --test_dataset "<DATASET_PATH>/GOPRO_Large/test/*/blur/*.*" --result <RESULT_PATH> --test
```

### DVD
```bash
python main-Pytorch.py --data_folder <MODEL_PATH> --test_dataset "<DATASET_PATH>/DVD_3840FPS_AVG_3-21/test/*/*.png" --result <RESULT_PATH> --test
```
### NFS
```bash
python main-Pytorch.py --data_folder <MODEL_PATH> --test_dataset "<DATASET_PATH>/NFS_3840FPS_AVG_3-21/test/*/*.png" --result <RESULT_PATH> --test
```
### HIDE
```bash
python main-Pytorch.py --data_folder <MODEL_PATH> --test_dataset "<DATASET_PATH>/HIDE_dataset/test/*/*.png" --result <RESULT_PATH> --test
```
### real_dataset
```bash
python main-Pytorch.py --data_folder <MODEL_PATH> --test_dataset "<DATASET_PATH>/real_dataset/*.jpg" --result <RESULT_PATH> --test
```

# Kernel Prediction
## Train
```bash
python main-Pytorch.py --gpu_id 0 --data_folder <MODEL_PATH> --model Kernel-Prediction --epoch 100 --pretrain 100 --dataset MSCOCO --dataset_train <DATASET_PATH>/MSCOCO/train_apply2 --image_size 100 --kernel_mode Estimate-Kernel-size --kernel_min 2 --kernel_max 20
```
## Predict
### GOPRO
```bash
python main-Pytorch.py --gpu_id 0 --data_folder <MODEL_PATH> --model Kernel-Prediction --test --test_dataset "<DATASET_PATH>/GOPRO_Large/train/*/*blur/*.*" --result <RESULT_PATH>/GOPRO_kernel  --image_size 100 --kernel_mode Estimate-Kernel-size --patch
```
### DVD
```bash
python main-Pytorch.py --gpu_id 0 --data_folder <MODEL_PATH> --model Kernel-Prediction --test --test_dataset "<DATASET_PATH>/DVD_3840FPS_AVG_3-21/train/blur/*/*.png" --result <RESULT_PATH>/DVD_kernel  --image_size 100 --kernel_mode Estimate-Kernel-size --patch
```
### NFS
```bash
python main-Pytorch.py --gpu_id 0 --data_folder <MODEL_PATH> --model Kernel-Prediction --test --test_dataset "<DATASET_PATH>/NFS_3840FPS_AVG_3-21/train/blur/*/*.png" --result <RESULT_PATH>/NFS_kernel  --image_size 100 --kernel_mode Estimate-Kernel-size --patch
```
### HIDE
```bash
python main-Pytorch.py --gpu_id 0 --data_folder <MODEL_PATH> --model Kernel-Prediction --test --test_dataset "<DATASET_PATH>/HIDE_dataset/train/*.png" --result <RESULT_PATH>/HIDE_kernel  --image_size 100 --kernel_mode Estimate-Kernel-size --patch
```

# Make Dataset
```bash
python make_dataset.py --mode Blur-Classification-All --save_path "<DATASET_PATH>/Kernel_dataset/Customized" --theory --average 14 --sigma 1 --min 0 --max 30
```
