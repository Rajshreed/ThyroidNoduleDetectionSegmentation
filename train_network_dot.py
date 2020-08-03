# Default to be a python3 script
import numpy as np
from glob import glob
import sys
import os
import random
from Unet_2 import get_unet2
from data_loaders import Generator_Thyroid_clickpt_roi_segmentation

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Basic configuration
#np.random.seed(1973) # for reproducibility
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# Set the number of rows and columns of the input image.
IMG_ROWS = 512
IMG_COLS = 512

BATCH_SIZE = 1 
# -----------------------------------
# Create the data generator
# -----------------------------------
# Map the labels
labels_dict = {}


# Get the dicom filenames
data_path = '../../data/TNSCUI2020_train/augmentation_task/train_unet/img_aug/'
mask_path = '../../data/TNSCUI2020_train/augmentation_task/train_unet/msk_aug/'

filenames = glob(data_path+'*')
#random.sample(filenames, len(filenames)) #shuffle(filenames)
random.shuffle(filenames)

# Transform the list of filenames into a numpy array
filenames = np.array(filenames)

#train_filenames, validation_filenames = np.split(filenames, [200000])
#print(len(train_filenames), len(validation_filenames))

# Create the actual generator
data_loader = Generator_Thyroid_clickpt_roi_segmentation(filenames, mask_path, new_size=(IMG_ROWS, IMG_COLS), labels_dict=labels_dict, batch_size=BATCH_SIZE)
#valid_data_loader = Generator_Thyroid_clickpt_roi_segmentation(validation_filenames, mask_path, new_size=(IMG_ROWS, IMG_COLS), labels_dict=labels_dict, batch_size=BATCH_SIZE)

# -------------------------------------
# Create and preload the network
# -------------------------------------
save_path = './models/'
save_name = 'unet2_v2'

if os.path.isdir(save_path):
	pass
else: 
	os.mkdir(save_path)



# Load the model with the previous weights
model = get_unet2(lr=1e-5, img_row=IMG_ROWS, img_cols=IMG_COLS,multigpu=0, channels_count=2)

model_json = model.to_json() # need to save to base model due to bugs with  keras multi-gpu
with open(save_path+save_name+".json", "w") as json_file:
        json_file.write(model_json)


model_save_path = save_path+save_name+"--{epoch:02d}-{loss:.4f}"
model_checkpoint = ModelCheckpoint(model_save_path + '.h5', monitor='loss',mode = 'min', save_best_only=False,period=1)
#early_stopping = EarlyStopping(monitor='loss', patience=5000, min_delta=1E-3)

# -------------------------------------
# Train and save the model
# -------------------------------------
epochs = 1000

model.fit_generator(data_loader.generate_data(), 
	#validation_data = valid_data_loader,
	steps_per_epoch = filenames.shape[0]/BATCH_SIZE, epochs=epochs, 
	callbacks=[model_checkpoint]) #, early_stopping])

model_json = model.to_json() # need to save to base model due to bugs with  keras multi-gpu
with open(save_path+save_name+".json", "w") as json_file:
	json_file.write(model_json)


