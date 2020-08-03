import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
input_size = (256,256) # (224,244)
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/train_imgmsk/",target_size=input_size)
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/valid_imgmsk/", target_size=input_size)

reg_param=0.0001


model = Sequential()
model.add(Conv2D(16,kernel_size=(7,7),strides=3, activation='relu', kernel_regularizer = l2(reg_param), input_shape=(input_size[0],input_size[1],3)))
#model.add(Dropout(0.5))
model.add(Conv2D(32,kernel_size=(5,5),strides=3, activation='relu',kernel_regularizer = l2(reg_param)))
#model.add(Dropout(0.5))
model.add(Conv2D(64,kernel_size=(9,9),strides=3, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.8))
model.add(Dense(50))
model.add(Dense(2,activation='softmax'))


print(model.summary())

from keras.optimizers import Adam 
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("./models/cnn_imgmsk_{epoch:02d}-{val_accuracy:.2f}.h5", monitor='acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='acc', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=1500, generator=traindata, validation_data= testdata, validation_steps=150, epochs=1000, callbacks=[checkpoint,early])



