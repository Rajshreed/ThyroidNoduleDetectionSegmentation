import numpy as np
import glob 
import os
import keras
import cv2
from keras.preprocessing import image
from keras import applications
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.applications.vgg16 import preprocess_input
import keras.backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

img_height = 200
img_width = 200
num_classes = 2

original_msk_path = "../../data/TNSCUI2020_train/mask/"

traindata_path = "../../data/TNSCUI2020_train/classification_ds/cropped_images_ds/train/no/"
#testdata_path = "../../data/TNSCUI2020_train/classification_ds/cropped_images_ds/valid/no/"

base_model = applications.vgg16.VGG16(weights= None, include_top=False, input_shape= (img_height,img_width,3)) # None

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

model.load_weights("./models/vgg16_cropped/vgg16_cropped268-0.80.h5") #("./models/vgg16_918-0.87.h5")
print(model.summary())
#exit(0)

layer_name = 'global_average_pooling2d_1' #'block5_pool'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)

from keras.optimizers import Adam #SGD#Adam
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#print(model.layers[18].name)

vgg16_feature_list = []
for filename in glob.glob(traindata_path + "*"):
	img = image.load_img(filename, target_size=(img_height, img_width,3))
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)
	vgg16_feature = intermediate_layer_model.predict(img_data)
	vgg16_feature_flat = np.squeeze(vgg16_feature).flatten()

	#find original height and width of the mask
#	pngname=filename.split('/')[-1][:4]
#	if pngname[1] == '_' or  pngname[1] == 'c':
#		pngname = pngname[:1]
#	elif pngname[2] == '_' or  pngname[2] == 'c':
#		pngname = pngname[:2]
#	elif pngname[3] == '_' or  pngname[3] == 'c':
#		pngname = pngname[:3]
#	else : #if pngname[4] == '_' or  pngname[4] == 'c':
#		pngname = pngname[:4]
	#print(filename, pngname)
#	mask_filename = original_msk_path + pngname+".PNG" #filename.split('/')[-1].split('.')[-3]+".PNG"
	mask_filename = original_msk_path + filename.split('/')[-1].split('.')[-3]+".PNG"

	msk_img = image.load_img(mask_filename)
	msk_img = image.img_to_array(msk_img)
	a = np.where(msk_img[:,:,0] != 0)
	bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
	#print((bbox[1]-bbox[0]), (bbox[3]-bbox[2]))
	#exit(0)
	
	vgg16_feature_flat = np.append(vgg16_feature_flat,[bbox[1]-bbox[0], bbox[3]-bbox[2], filename.split('/')[-1]]) 
	vgg16_feature_list.append(vgg16_feature_flat)
	#print(filename, vgg16_feature_flat[-3:])
	#print(vgg16_feature_flat, vgg16_feature_flat.shape)
	#for x in range(len(vgg16_feature_flat)): 
	#	print(vgg16_feature_flat[x]) 
	#exit(0)
#print(vgg16_feature_list, len(vgg16_feature_list))
np.save("../../data/TNSCUI2020_train/classification_ds/vgg16_cropped512features_train_no.npy", vgg16_feature_list)


