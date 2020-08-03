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

test_path_no = "../../data/TNSCUI2020_train/classification_ds/resized_test/no/"
test_path_yes = "../../data/TNSCUI2020_train/classification_ds/resized_test/yes/"

x_test = []
y_test = []
for filename in glob.glob(test_path_no+'*'):
    image = cv2.imread(filename)
    image = np.asarray(image)
    image = cv2.resize(image, (224,224))
    image = np.rollaxis(image, 2,0)
    x_test.append(image)
    y_test.append([1,0]) # first element = no, second element = yes

for filename in glob.glob(test_path_yes+'*'):
    image = cv2.imread(filename)
    image = np.asarray(image)
    image = cv2.resize(image, (224,224))
    image = np.rollaxis(image, 2, 0)
    x_test.append(image)
    y_test.append([0,1]) # first element = no, second element = yes

x_test = np.array(x_test)
y_test = np.array(y_test)

model = create_googlenet()

from keras.optimizers import SGD#Adam
opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.load_weights('./models/googlenet_Epoch150TrainLoss2.3071635ValLoss0.6850274model.h5')

out = model.predict(x_test)
#print(out[2])
print("validation loss = ", np.sum( np.abs( np.subtract( np.argmax(out[2],axis=1), np.argmax(y_test,axis=1))))/(len(y_test)) )



