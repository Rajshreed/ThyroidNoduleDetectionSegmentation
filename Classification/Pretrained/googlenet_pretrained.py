import numpy as np
import glob 
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.regularizers import l2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

img_height = 200
img_width = 200
num_classes = 2

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/cropped_images_ds/train/",target_size=(200,200))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/cropped_images_ds/valid/", target_size=(200,200))


base_model = applications.InceptionV3(weights= 'imagenet', include_top=False, input_shape= (img_height,img_width,3)) # 'imagenet'

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu', name='1024featuresgooglenet', kernel_regularizer=l2(0.00001))(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

# Freeze the layers except the last 3 layers
for layer in model.layers[:-3]:
    layer.trainable = False

print(model.summary())
exit(0)
from keras.optimizers import Adam #SGD#Adam
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("./models/inceptionV3_imagenet{epoch:02d}-{val_accuracy:.2f}.h5", monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=200,generator=traindata, validation_data= testdata, validation_steps=10,epochs=1000,callbacks=[checkpoint,early])
# steps per epoch=100 -> 20000
