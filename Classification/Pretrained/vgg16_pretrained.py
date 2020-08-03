import numpy as np
import glob 
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras import optimizers
from keras.models import Sequential,Model,load_model
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

img_height = 200#224
img_width = 200#224
num_classes = 2

def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join('./models/', 'tmp_vgg16_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/cropped_images_ds/train/",target_size=(200,200))#224,224
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="../../data/TNSCUI2020_train/classification_ds/cropped_images_ds/valid/", target_size=(200,200))


base_model = applications.vgg16.VGG16(weights= None, include_top=False, input_shape= (img_height,img_width,3)) # None

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

add_regularization(model)

print(model.summary())
#exit(0)

from keras.optimizers import Adam #SGD#Adam
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("./models/vgg16_cropped{epoch:02d}-{val_accuracy:.2f}.h5", monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto')

hist = model.fit_generator(steps_per_epoch=20000,generator=traindata, validation_data= testdata, validation_steps=10,epochs=1000,callbacks=[checkpoint,early])

