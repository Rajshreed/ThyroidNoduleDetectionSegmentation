import numpy as np
import cv2
import glob
import random
from create_crop_aug_dataset import *

label_ds = np.loadtxt('train.csv', dtype='str', delimiter=',')

#remove the header row 
label_ds = label_ds[1:]

#randomly shuffle the sample
label_ds = random.sample(list(label_ds), len(label_ds))

test_yes_count=0
test_no_count=0
valid_yes_count=0
valid_no_count=0

for i in range(len(label_ds)):
    filename = (label_ds[i][0]).split('/')[-1]
    #print(filename)
    msk_filename = './mask/' + filename
    img_filename = './image/' + filename
    mask = cv2.imread(msk_filename)
    image = cv2.imread(img_filename)


    label = label_ds[i][1]

    print(filename, label)

    if test_yes_count < 400 and label=='1' : #r valid_no_count < 300:
        base_path = './classification_ds/cropped_images_ds/test/yes/'
        test_yes_count+=1
        #print("copy test 1",i, filename, label) #move test - 1
    elif test_no_count < 328 and label=='0' :
        base_path = './classification_ds/cropped_images_ds/test/no/'
        test_no_count+=1
        #print("copy test 0",i, filename, label) #move test - 0
    elif valid_yes_count < 400 and label=='1' : #r valid_no_count < 300:
        base_path = './classification_ds/cropped_images_ds/valid/yes/'+filename
        valid_yes_count+=1
        #print("copy valid 1",i, filename, label) #move valid - 1
    elif valid_no_count < 328 and label=='0' : 
        base_path = './classification_ds/cropped_images_ds/valid/no/'+filename
        valid_no_count+=1
        #print("copy valid 0",i, filename, label) #move valid - 0
    elif label == '1':
        base_path = './classification_ds/cropped_images_ds/train/yes/'+filename
        #print("copy train 1", i, filename, label) #move train - 1
    else:
        base_path = './classification_ds/cropped_images_ds/train/no/'+filename
        #print("copy train 0", i, filename, label)#move train - 0

    create_cropped_images(filename, image[:,:,0], mask[:,:,0], base_path)

    print("i=",i, "test_yes_count=",test_yes_count, "test_no_count=",test_no_count, "valid_yes_count=",valid_yes_count, "valid_no_count=", valid_no_count, base_path, image.shape)

