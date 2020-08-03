'''
Description:
	predict python script for UNet model
'''

from glob import glob
import numpy as np
import cv2
from Unet_0 import get_unet0
from data_loaders import Generator_Thyroid_clickpt_roi_segmentation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

# Set the number of rows and columns of the input image.
IMG_ROWS = 512#512
IMG_COLS = 512#512


# Basic configuration
np.random.seed(1973) # for reproducibility
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


image_path = '../../data/TNSCUI2020_train/augmentation_task/valid_unet/img/'
image_masks_path = '../../data/TNSCUI2020_train/augmentation_task/valid_unet/msk/'

# # Identify all the images in the test set
filenames = glob(image_path + '*')
filenames = np.array(filenames)
# -----------------------------------
# Create the data generator
# -----------------------------------
# Map the labels
labels_dict = {}

label_path = ''
data_generator = Generator_Thyroid_clickpt_roi_segmentation(filenames, label_path, (IMG_ROWS, IMG_COLS), labels_dict)
# -------------------------------------
# Create and preload the network
# -------------------------------------

weights_path = './models/unet0_medium_v2--100--0.82.h5' #'./models/unet0_small_v1--167--0.91.h5' #'./models/unet2_large_v1--82--0.91.h5' #./models/unet2_v2--04--0.8116.h5'
model,_  = get_unet0(lr=1e-5, img_row=IMG_ROWS, img_cols=IMG_COLS, multigpu=0, channels_count=2)
model.load_weights(weights_path)

# -------------------------------------
# Make the predictions
# -------------------------------------
cnt = 0

for file in filenames:
	# Load the image
	img_input = cv2.imread(file)
	jpg_name = file.split('/')[-1]
	mask = cv2.imread(image_masks_path+jpg_name)
	#img_input = img_input[50:550, 50:750, :]
	#mask = mask[50:550,50:750,:]	
#	img_small = np.zeros((500,700,3),np.uint8)
#	msk_small = np.zeros((500,700,3),np.uint8)
#	img_s = cv2.resize(img_input,(350,250))
#	msk_s = cv2.resize(mask,(350,250))
#	img_small[125:375,175:525,0] = img_s[:,:,0]
#	img_small[125:375,175:525,1] = img_s[:,:,1]
#	msk_small[125:375,175:525,:] = msk_s
#	img_input = img_small
#	mask = msk_small
	
#	img_input[:,:,1] = cv2.dilate(img_input[:,:,1], np.ones((3,3), np.uint8) )
#	cv2.imshow("small_img", img_input)
#	cv2.waitKey(5000)

	X, _ = data_generator.preprocess_data(img_input[:,:,:2])
	X = np.array([X])
	prediction = model.predict(X)
	prediction = (prediction[0,:,:,0] > 0.5)
	#print(img_input.shape)
	#exit(0)
	prediction1=np.uint8(prediction)*255
	prediction1 = cv2.resize(prediction1,(700,500))
	show = np.zeros((500,700,3),np.uint8)
	show[:,:,0] = img_input[:,:,0]#[50:550,50:750,0]
	show[:,:,1] = mask[:,:,0]#[50:550,50:750,0]
	show[:,:,2] = prediction1
	path_='../../data/TNSCUI2020_train/augmentation_task/results_unet_predict_mediumunet/' + jpg_name
	cv2.imwrite(path_, show)
	#cv2.imshow("show", show)
	#cv2.imshow("img_input", img_input[:,:,0])
	#cv2.waitKey(5000)
	#break
# Prediction provided by Unet, needs to be post processed to include only blob part of the dot and dilate+erode (also called opening) to make it more round

#	contours, _ = cv2.findContours( prediction1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#	found = 0
#	for c in contours:
#		blob = np.zeros((600,800,3), np.uint8)
#		cv2.drawContours(blob, [c], -1, (255,255,255), -1)
		#cv2.imshow("contour", blob)
		#cv2.waitKey(5000)
#		if (np.sum(np.multiply(blob[:,:,0], img_input[:,:,1])) != 0):
#			found=1
#			print("Found")
#			break
#	if (found == 1):
#		bb_1 = cv2.boundingRect(np.uint8(blob[:,:,0]))
#		current_blob_size = int ( (bb_1[2] + bb_1[3])/2)
#		if (current_blob_size <= 10):
#			current_blob_size=10
#		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.2*current_blob_size), int(0.2*current_blob_size)))
# np.ones((, int(0.2*current_blob_size)), dtype=np.uint8)
#		prediction1_smooth = cv2.morphologyEx(prediction1, cv2.MORPH_CLOSE, kernel)
#		prediction1_smooth = cv2.morphologyEx(prediction1, cv2.MORPH_OPEN, kernel, iterations=2)

#		bb = cv2.boundingRect(prediction1_smooth)
##		cv2.rectangle(prediction1_smooth, (bb[0]-avg, bb[1]-avg),(bb[0]+bb[2]+avg, bb[1]+bb[3]+avg), (255,255,255), 2)
#		show1 = np.zeros((600,800,3),np.uint8)
#		show1[:,:,0] = img_input[:,:,0]
#		show1[:,:,1] = mask[:,:,0]
#		show1[:,:,2] = prediction1_smooth
		#cv2.imshow("show1", show1)
		#cv2.waitKey(5000)
		
#		cv2.imshow("cropped", img_input[max(bb[1]-avg,0) : min(bb[1]+bb[3]+avg, 600), max(bb[0]-avg,0) :min(bb[0]+bb[2]+avg, 800), 0])
		#cv2.waitKey(5000)
#		cv2.imshow("cropped_msk", mask[max(bb[1]-avg,0) : min(bb[1]+bb[3]+avg, 600), max(bb[0]-avg,0) :min(bb[0]+bb[2]+avg, 800), 0])
#		cv2.imwrite("/data/Atefeh/Rajshree/ROIs_0/"+jpg_name, img_input[max(bb[1]-avg,0) : min(bb[1]+bb[3]+avg, 600), max(bb[0]-avg,0) :min(bb[0]+bb[2]+avg, 800), 0])
#		cv2.imwrite("/data/Atefeh/Rajshree/Masks_0/"+jpg_name,  mask[max(bb[1]-avg,0) : min(bb[1]+bb[3]+avg, 600), max(bb[0]-avg,0) :min(bb[0]+bb[2]+avg, 800), 0])
#		cv2.waitKey(5000)

#	(x,y,w,h)=cv2.boundingRect(prediction1)
#	img_input_s = img_input[y:y+h, x:x+w]
#	mask_s = mask[y:y+h, x:x+w]
#	print((x,y,w,h))
#	print(img_input_s.shape)
#	X, _ = data_generator.preprocess_data(img_input_s[:,:,:2])
#	X = np.array([X])
#	prediction = model.predict(X)
#	prediction = (prediction[0,:,:,0] > 0.5)
	
#	show = np.zeros((h,w,3),np.uint8)
#	show[:,:,0] = img_input[y:y+h, x:x+w,0]#img_input_s[:,:,0]
#	show[:,:,1] = mask[y:y+h, x:x+w,0]#mask_s[:,:,0]
#	show[:,:,2] = cv2.resize(np.uint8(prediction)*255, (w, h))
#	cv2.imshow("show", show)
#	cv2.waitKey(5000)
#	img_input=None
#	mask=None
#	cnt = cnt + 1
#	if cnt == 100:
#		break

