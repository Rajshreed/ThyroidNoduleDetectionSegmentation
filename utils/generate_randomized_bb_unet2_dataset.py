import numpy as np
import glob
import cv2
import random

#validation paths
#combined_img_msk_path = './valid_img_msk/'
#img_save_base_path = './valid_unet/img/'
#msk_save_base_path = './valid_unet/msk/'

#training paths
combined_img_msk_path = './img_msk_aug/'
img_save_base_path = './train_unet/img_aug/'
msk_save_base_path = './train_unet/msk_aug/' 

for file_img in glob.glob(combined_img_msk_path + '*'):
    img_input = cv2.imread(file_img)
    msk = np.zeros([500,700,3], np.uint8)
    msk[:,:,0] = img_input[:,:,1] 
    img_input = img_input[:,:,0]
    print(img_input.shape,msk.shape)
    bb = cv2.boundingRect(msk[:,:,0]) 
    box1 = np.zeros([500,700],np.uint8)
    box2 = np.zeros([500,700],np.uint8)
    box3 = np.zeros([500,700],np.uint8)
    img = np.zeros([500,700,3], np.uint8)
    img[:,:,0] = img_input
    #img[:,:,2] = box1 #msk[:,:,0] #box
    r_bb2 = int(0.20*bb[2]) 
    r_bb3 = int(0.20*bb[3]) 
    r1 = random.randint(0, r_bb3) + random.randint(0,20)
    r2 = random.randint(0, r_bb3) + random.randint(0,20)
    r3 = random.randint(0, r_bb2) + random.randint(0,20)
    r4 = random.randint(0, r_bb2) + random.randint(0,20)
    print(r1,r2,r3,r4)
 
    box1[max(bb[1]-r1,0):min(bb[1]+bb[3]+r2,500), max(0,bb[0]-r3): min(700, bb[0]+bb[2]+r4)] = 255
    img[:,:,1] = box1
    
    #cv2.imshow("img",img)
    #cv2.imshow("msk",msk)
    #cv2.waitKey(1000)

    save_path = img_save_base_path + file_img.split('/')[-1].split('.')[-2]+"_bb1.png" 
    cv2.imwrite(save_path, img)   
    save_path_msk = msk_save_base_path + file_img.split('/')[-1].split('.')[-2]+"_bb1.png" 
    cv2.imwrite(save_path_msk, msk)
    
    r1 = random.randint(0, r_bb3) + random.randint(0,20)
    r2 = random.randint(0, r_bb3) + random.randint(0,20)
    r3 = random.randint(0, r_bb2) + random.randint(0,20)
    r4 = random.randint(0, r_bb2) + random.randint(0,20)
    box2[max(bb[1]-r1,0):min(bb[1]+bb[3]+r2,500), max(0,bb[0]-r3): min(700, bb[0]+bb[2]+r4)] = 255  
    save_path = img_save_base_path + file_img.split('/')[-1].split('.')[-2]+"_bb2.png" 
    img[:,:,1]  = box2
    cv2.imwrite(save_path, img)   
    save_path_msk = msk_save_base_path + file_img.split('/')[-1].split('.')[-2]+"_bb2.png" 
    cv2.imwrite(save_path_msk, msk)   
 
    box3[max(bb[1],0):min(bb[1]+bb[3],500), max(0,bb[0]): min(700, bb[0]+bb[2])] = 255  
    save_path = img_save_base_path + file_img.split('/')[-1].split('.')[-2]+"_bb3.png" 
    img[:,:,1]  = box3
    cv2.imwrite(save_path, img)   
    save_path_msk = msk_save_base_path + file_img.split('/')[-1].split('.')[-2]+"_bb3.png" 
    cv2.imwrite(save_path_msk, msk)   

