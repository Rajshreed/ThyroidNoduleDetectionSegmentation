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

#traindata_path = "../../data/TNSCUI2020_train/classification_ds/resized_train/no/"
#validdata_path = "../../data/TNSCUI2020_train/classification_ds/resized_valid/no/"
testdata_path = "../../data/TNSCUI2020_train/classification_ds/cropped_images_ds/test/yes/"

base_model = applications.vgg16.VGG16(weights= None, include_top=False, input_shape= (img_height,img_width,3)) # None

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

model.load_weights('./models/vgg16_cropped/vgg16_cropped268-0.80.h5') #("./models/vgg16_918-0.87.h5")

from keras.optimizers import Adam #SGD#Adam
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
pred=[]
for filename in glob.glob(testdata_path + "*"):
	img = image.load_img(filename, target_size=(img_height, img_width,3))
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)
	#p = np.argmax(model.predict(img_data))
	p = model.predict(img_data)
	p_argmax = np.argmax(p)
	pred = np.append(pred, p)
	print(filename.split('/')[-1], p, p_argmax)

print("accuracy = ", np.sum(pred==0)/len(pred))

