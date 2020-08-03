import numpy as np
import cv2
import glob
import random

#train_ds = np.loadtxt('../../source/keras-retinanet-master/keras_retinanet/bin/train_csvinput.csv', dtype='str', delimiter=',')
#valid_ds = np.loadtxt('../../source/keras-retinanet-master/keras_retinanet/bin/valid_csvinput.csv', dtype='str', delimiter=',')
label_ds = np.loadtxt('train.csv', dtype='str', delimiter=',')
label_ds = label_ds[1:]
random.sample(list(label_ds), len(label_ds))
valid_yes_count=0
valid_no_count=0

max_x=0
max_y=0

#ds = valid_ds # train_ds for training data and valid_ds for validation data
for i in range(len(label_ds)):
    filename = (label_ds[i][0]).split('/')[-1]
    #print(filename)
    msk_filename = './mask/' + filename
    img_filename = './image/' + filename
    mask = cv2.imread(msk_filename)
    image = cv2.imread(img_filename)
    bb = cv2.boundingRect(mask[:,:,0])

    r_bb2 = int(0.20*bb[2])
    r_bb3 = int(0.20*bb[3])
    r1 = r_bb3
    r2 = r_bb3
    r3 = r_bb2
    r4 = r_bb2
    #print(r1,r2,r3,r4)
    #print(image.shape)
    #box = np.zeros(image.shape, np.uint8)
    #box[max(bb[1]-r1,0):min(bb[1]+bb[3]+r2,image.shape[0]), max(0,bb[0]-r3): min(image.shape[1], bb[0]+bb[2]+r4),:] = 1
    #image = np.uint8(np.multiply(image,box))


    y1 = max(bb[1]-r1,0)
    y2 = min(bb[1]+bb[3]+r2,image.shape[0]) 
    x1 = max(0,bb[0]-r3)
    x2 = min(image.shape[1], bb[0]+bb[2]+r4)

    box = image[y1:y2, x1:x2,:] #box = np.zeros([abs(y2-y1)+1,abs(x2-x1)+1,3], np.uint8)
    max_x = max(x2-x1,max_x) 
    max_y = max(y2-y1,max_y) 
    label = label_ds[i][1]

    #print(filename, label, box.shape)
    if valid_yes_count < 300 and label=='1' : #r valid_no_count < 300:
        path = './classification_ds/valid/yes/'+filename
        valid_yes_count+=1
            #print("copy valid 1",i, filename, label) #move valid - 1
    elif valid_no_count < 300 and label=='0' : 
        path = './classification_ds/valid/no/'+filename
        valid_no_count+=1
            #print("copy valid 0",i, filename, label) #move valid - 0
    elif label == '1':
        path = './classification_ds/train/yes/'+filename
            #print("copy train 1", i, filename, label) #move train - 1
    else:
        path = './classification_ds/train/no/'+filename
            #print("copy train 0", i, filename, label)#move train - 0
    #print(i, valid_yes_count, valid_no_count, path, image.shape)
    #print(cv2.imwrite(path, box))
#    print(filename, label_ds[(np.where(label_ds[:,0] == filename)[0][0]), 1] )
print("max_x=",max_x,"max_y=", max_y) 
