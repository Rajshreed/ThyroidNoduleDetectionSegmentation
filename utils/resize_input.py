import numpy as np
import glob 
import cv2

for file1 in glob.glob('./mask/*'):
    image = cv2.imread(file1)
    resized = cv2.resize(image, (700,500), cv2.INTER_LINEAR)
    save_path = './resized_mask/'+file1.split('/')[-1]
    #print(image.shape, resized.shape, save_path)
    cv2.imwrite(save_path, resized)

