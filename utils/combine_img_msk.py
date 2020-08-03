import numpy as np
import glob 
import cv2
cnt = 0
train_ds = np.loadtxt('../../source/keras-retinanet-master/keras_retinanet/bin/train_csvinput.csv', dtype='str', delimiter=',')
valid_ds = np.loadtxt('../../source/keras-retinanet-master/keras_retinanet/bin/valid_csvinput.csv', dtype='str', delimiter=',')
#label_ds = np.loadtxt('train.csv', dtype='str', delimiter=',')

ds = valid_ds # train_ds for training data and valid_ds for validation data

for i in range(len(ds)): 
    filename = (ds[i][0]).split('/')[-1]
#    print(filename, label_ds[(np.where(label_ds[:,0] == filename)[0][0]), 1] )
#exit(0)

#for file1 in glob.glob('./resized_mask/*'):
    mask = cv2.imread('./resized_mask/' + filename)
    image = cv2.imread('./resized_image/'+filename)
    combined = np.zeros(image.shape, np.uint8)
    combined[:,:,0] = image[:,:,0]
    combined[:,:,1] = mask[:,:,0]
    cnt = cnt+1
    save_path = './augmentation_task/valid_img_msk/'+filename
    #print(image.shape, mask.shape, combined.shape, save_path)
    cv2.imwrite(save_path, combined)
    print(cnt)
