import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/resized_train/",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/resized_valid/", target_size=(224,224))

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#model.add(Conv2D(filters=1024, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(Conv2D(filters=1024, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(Conv2D(filters=1024, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#model.add(Conv2D(filters=2048, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(Conv2D(filters=2048, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(Conv2D(filters=2048, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Flatten())
#model.add(Dense(units=16384,activation="relu"))#4096
#model.add(Dense(units=8192,activation="relu"))
model.add(Dense(units=4096,activation="relu"))#4096
model.add(Dense(units=1024,activation="relu"))#4096
model.add(Dense(units=2, activation="softmax"))
print(model.summary())

from keras.optimizers import Adam #SGD#Adam
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("./models/vgg16_{epoch:02d}-{val_accuracy:.2f}.h5", monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=1000,callbacks=[checkpoint,early])



