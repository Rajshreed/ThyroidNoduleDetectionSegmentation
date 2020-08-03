# Default to be a python3 script
import numpy as np
from glob import glob
import sys
import os
import random
from Unet_2 import get_unet2
from data_loaders import Generator_Thyroid_clickpt_roi_segmentation
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import shutil

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Basic configuration
#np.random.seed(1973) # for reproducibility
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# Set the number of rows and columns of the input image.
IMG_ROWS = 512
IMG_COLS = 512

BATCH_SIZE = 1 

# Get the dicom filenames
data_path = '../../data/TNSCUI2020_train/augmentation_task/train_unet/img_aug/'
mask_path = '../../data/TNSCUI2020_train/augmentation_task/train_unet/msk_aug/'
save_path_img_sml = '../../data/TNSCUI2020_train/augmentation_task/segregated_aug_data/img_sml/'
save_path_msk_sml = '../../data/TNSCUI2020_train/augmentation_task/segregated_aug_data/msk_sml/'
save_path_img_med = '../../data/TNSCUI2020_train/augmentation_task/segregated_aug_data/img_med/'
save_path_msk_med = '../../data/TNSCUI2020_train/augmentation_task/segregated_aug_data/msk_med/'
save_path_img_lrg = '../../data/TNSCUI2020_train/augmentation_task/segregated_aug_data/img_lrg/'
save_path_msk_lrg = '../../data/TNSCUI2020_train/augmentation_task/segregated_aug_data/msk_lrg/'

filenames = glob(data_path+'*')

# Transform the list of filenames into a numpy array
filenames = np.array(filenames)
min_h=10000
min_w=10000
max_h=0
max_w=0

cnt_sml=0
cnt_med=0
cnt_lrg=0

for filename in filenames:
    img = cv2.imread(filename)
    msk = img[:,:,1]
    png_name = filename.split('/')[-1]
    pnts = np.where(msk >0)
    bbox = min(pnts[0]), max(pnts[0]), min(pnts[1]), max(pnts[1]) # x1, x2, y1, y2
    w=bbox[1]-bbox[0]
    h=bbox[3]-bbox[2]
    if (w<200 and h<200):
        cnt_sml+=1
        shutil.copy( filename, save_path_img_sml + png_name )
        shutil.copy( mask_path+ png_name, save_path_msk_sml + png_name )

    elif (w<400 and h<400):
        cnt_med+=1
        shutil.copy( filename, save_path_img_med + png_name )
        shutil.copy( mask_path+ png_name, save_path_msk_med + png_name )

    else: 
        cnt_lrg+=1
        shutil.copy( filename, save_path_img_lrg + png_name )
        shutil.copy( mask_path+ png_name, save_path_msk_lrg + png_name )

    #print("cnt small=", cnt_sml, "cnt med=", cnt_med, "cnt_lrg=", cnt_lrg)
    #print( filename, "width=", bbox[1]-bbox[0], "height=", bbox[3]-bbox[2] )
    #min_h = min(min_h, bbox[3]-bbox[2])
    #min_w = min(min_w, bbox[1]-bbox[0])
    #max_h = max(max_h, bbox[3]-bbox[2])
    #max_w = max(max_w, bbox[1]-bbox[0])
#print("(min_h, min_w, max_h, max_w) =", min_h, min_w, max_h, max_w)
#print("cnt small=", cnt_sml, "cnt med=", cnt_med, "cnt_lrg=", cnt_lrg)
