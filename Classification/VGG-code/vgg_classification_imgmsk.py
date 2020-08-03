import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/train_imgmsk/",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/valid_imgmsk/", target_size=(224,224))

reg_param=0.000001

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_regularizer=l2(reg_param), bias_regularizer=l2(reg_param)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.8))
model.add(Dense(units=1024,activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.9))
model.add(Dense(units=2, activation="softmax"))
print(model.summary())

from keras.optimizers import Adam 
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("./models/vgg16_reg6_weighted_imgmsk_{epoch:02d}-{val_accuracy:.2f}.h5", monitor='acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='acc', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=1500,generator=traindata, validation_data= testdata, validation_steps=150,epochs=1000,callbacks=[checkpoint,early], class_weight=[6,5])



