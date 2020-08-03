import numpy as np
import glob
import cv2
from keras.models import load_model
saved_model = load_model("./models/vgg16_reg_weighted_imgmsk_08-0.79.h5") #("./models/v2/vgg16_445-0.80.h5")

count_no=0
count_yes=0
for filename in glob.glob('../../data/TNSCUI2020_train/classification_ds/test_imgmsk/no/*'):
    img = cv2.imread(filename)
    img = np.asarray(img)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)

    output = saved_model.predict(img)
    if output[0][0] > output[0][1]:
        #print("no")
        count_no+=1
        #print(filename.split('/')[-1])
    else:
        #print("yes")
        count_yes+=1
        #print(filename.split('/')[-1])

print("No=",count_no)
print("Yes=", count_yes)
