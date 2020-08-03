import keras,os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
import cv2
from googlenet import create_googlenet
from keras import backend

# force channels-first ordering
backend.set_image_data_format('channels_first')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

train_path_no = "../../data/TNSCUI2020_train/classification_ds/resized_train/no/"
train_path_yes = "../../data/TNSCUI2020_train/classification_ds/resized_train/yes/"

valid_path_no = "../../data/TNSCUI2020_train/classification_ds/resized_valid/no/"
valid_path_yes = "../../data/TNSCUI2020_train/classification_ds/resized_valid/yes/"

BATCH_SIZE=32

x_train = []
y_train = []
for filename in glob.glob(train_path_no+'*'):
    image = cv2.imread(filename)
    image = np.asarray(image)
    image = cv2.resize(image, (224,224))
    image = np.rollaxis(image, 2,0)
    x_train.append(image)
    y_train.append([1,0]) # first element = no, second element = yes

for filename in glob.glob(train_path_yes+'*'):
    image = cv2.imread(filename)
    image = np.asarray(image)
    image = cv2.resize(image, (224,224))
    image = np.rollaxis(image, 2, 0)
    x_train.append(image)
    y_train.append([0,1]) # first element = no, second element = yes

x_train = np.array(x_train)
y_train = np.array(y_train)

x_valid = []
y_valid = []
for filename in glob.glob(valid_path_no+'*'):
    image = cv2.imread(filename)
    image = np.asarray(image)
    image = cv2.resize(image, (224,224))
    image = np.rollaxis(image, 2, 0)
    x_valid.append(image)
    y_valid.append([1,0]) # 0 = no, 1 = yes

for filename in glob.glob(valid_path_yes+'*'):
    image = cv2.imread(filename)
    image = np.asarray(image)
    image = cv2.resize(image, (224,224))
    image = np.rollaxis(image, 2, 0)
    x_valid.append(image)
    y_valid.append([0,1]) # 0 = no, 1 = yes

x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

model = create_googlenet()

from keras.optimizers import SGD#Adam
opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#from keras.callbacks import ModelCheckpoint, EarlyStopping
#checkpoint = ModelCheckpoint("./models/googlenet_{epoch:02d}-{val_accuracy:.2f}.h5", monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

datagen = ImageDataGenerator()
n_epochs= 100000

for e in range(n_epochs): 
    batches=0
    model.reset_metrics()
    for X_batch, Y_batch in datagen.flow(x_train, y_train, batch_size=BATCH_SIZE):
        loss = model.train_on_batch(X_batch, [Y_batch,Y_batch,Y_batch]) 
        metrics_names = model.metrics_names
        #print(metrics_names)
        print("train: ",
          "{}: {:.3f}".format(metrics_names[0], loss[0]),
          "{}: {:.3f}".format(metrics_names[3], loss[3]))

        batches += 1
        if batches >= len(x_train) / BATCH_SIZE:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    print("Epoch =", e, " completed\n")
    b=0
    for X_batch, Y_batch in datagen.flow(x_valid, y_valid, batch_size=BATCH_SIZE):
        result = model.test_on_batch(X_batch, [Y_batch,Y_batch,Y_batch],
                                 # return accumulated metrics
                                 reset_metrics=False)
        b+=1
        metrics_names = model.metrics_names
        print("\neval: ",
        "{}: {:.3f}".format(metrics_names[0], result[0]),
        "{}: {:.3f}".format(metrics_names[3], result[3]))
        if b >= len(x_valid) / BATCH_SIZE:
            break
    if e%5==0:
        # serialize model to JSON
        model_json = model.to_json()
        with open("./models/googlenet_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("./models/googlenet_"+"Epoch"+str(e)+"TrainLoss"+str(loss[0])+"ValLoss"+str(result[3])+"model.h5")

    out = model.predict(x_valid)
    #print(out[2])
    print("validation loss %= ", np.sum( np.abs( np.subtract( np.argmax(out[2],axis=1), np.argmax(y_valid,axis=1))))/(len(y_valid)) )



