import numpy as np
import cv2
import glob
import os

# function to get cropped image from original image with given center point and the crop size around it
def get_cropped_image(input_image, center_point, crop_size=(200,200)):# center_point = (c_pnt_y, c_pnt_x) and crop_size = (200,200)
	new_y = input_image.shape[0]
	new_x = input_image.shape[1]
	if (input_image.shape[0] < crop_size[0]): 
		new_y = crop_size[0]
	if (input_image.shape[1] < crop_size[1]):
		new_x = crop_size[1]
	canvas = np.zeros((new_y, new_x), np.uint8)
	canvas[0:input_image.shape[0], 0:input_image.shape[1]] = input_image	
	input_image = canvas
	y1 = center_point[0] - int(crop_size[0]/2)
	y2 = center_point[0] + int(crop_size[0]/2)

	if (y1 < 0):
		y1 = 0
		y2 = crop_size[0]
	elif (y2 > input_image.shape[0]):
		y1 = input_image.shape[0] - crop_size[0]
		y2 = input_image.shape[0]
	x1 = center_point[1] - int(crop_size[1]/2)
	x2 = center_point[1] + int(crop_size[1]/2)
	if (x1 < 0):
		x1 = 0
		x2 = crop_size[1]
	elif (x2 > input_image.shape[1]):
		x1 = input_image.shape[1] - crop_size[1]
		x2 = input_image.shape[1]

	return input_image[y1:y2, x1:x2]

org_img_path = "../../data/TNSCUI2020_train/image/"
org_msk_path = "../../data/TNSCUI2020_train/mask/"
count=0

for filename in glob.glob(org_img_path + "*"):
	png_name = filename.split('/')[-1]
	img = cv2.imread(filename)
	msk = cv2.imread(org_msk_path+png_name)
	img = img[:,:,0]
	msk = msk[:,:,0]
	pnts = np.where(msk>0)# rows, cols
	#bbox = min(pnts[0]), max(pnts[0]), min(pnts[1]), max(pnts[1])# y1, y2, x1, x2
	y1, y2, x1, x2 = min(pnts[0]), max(pnts[0]), min(pnts[1]), max(pnts[1])# y1, y2, x1, x2
	nodule_width = x2-x1
	nodule_height = y2-y1
	cp = (int(y1 + (y2-y1)/2), int( x1 + (x2-x1)/2) )# cp_y, cp_x
	# note all coordinates are in format (y,x) and not (x,y)
	corner1 = (y1, x1)
	corner2 = (y1, x2)
	corner3 = (y2, x2)
	corner4 = (y2, x1)
	mid_p1 = (y1, cp[1])
	mid_p2 = (y2, cp[1])
	mid_p3 = (cp[0], x1)
	mid_p4 = (cp[0], x2)
	x_1, x_2, y_1, y_2 = corner1[1], cp[1], corner1[0], cp[0]
	extra_mid_p1 = (int(y_1 + (y_2-y_1)/2), int( x_1 + (x_2-x_1)/2) ) # center point of bbox corner1, cp
	x_1, x_2, y_1, y_2 = mid_p1[1], mid_p4[1], mid_p1[0], mid_p4[0]
	extra_mid_p2 = (int(y_1 + (y_2-y_1)/2), int( x_1 + (x_2-x_1)/2) ) # center point of bbox mid_p1, mid_p4
	x_1, x_2, y_1, y_2 = cp[1], corner3[1], cp[0], corner3[0]
	extra_mid_p3 = (int(y_1 + (y_2-y_1)/2), int( x_1 + (x_2-x_1)/2) ) # center point of bbox cp, corner3
	x_1, x_2, y_1, y_2 = mid_p3[1], mid_p2[1], mid_p3[0], mid_p2[0]
	extra_mid_p4 = (int(y_1 + (y_2-y_1)/2), int( x_1 + (x_2-x_1)/2) ) # center point of bbox mid_p3, mid_p2

	# crop 9 sub-images 
	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_corner1"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, corner1, (200,200)))

	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_corner2"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, corner2, (200,200)))

	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_corner3"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, corner3, (200,200)))

	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"corner4"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, corner4, (200,200)))

	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_mid_p1"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, mid_p1, (200,200)))

	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_mid_p2"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, mid_p2, (200,200)))

	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_mid_p3"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, mid_p3, (200,200)))

	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_mid_p4"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, mid_p4, (200,200)))

	save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_cp"+".PNG"
	cv2.imwrite(save_path, get_cropped_image(img, cp, (200,200)))

	if nodule_width > 400 or nodule_height > 400 :
	#crop additional 4 sub-images for large nodules
		save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_extra_mid_p1"+".PNG"
		cv2.imwrite(save_path, get_cropped_image(img, extra_mid_p1, (200,200)))

		save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_extra_mid_p2"+".PNG"
		cv2.imwrite(save_path, get_cropped_image(img, extra_mid_p2, (200,200)))

		save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_extra_mid_p3"+".PNG"
		cv2.imwrite(save_path, get_cropped_image(img, extra_mid_p3, (200,200)))

		save_path = "../../data/TNSCUI2020_train/cropped_image/"+png_name.split('.')[-2]+"_extra_mid_p4"+".PNG"
		cv2.imwrite(save_path, get_cropped_image(img, extra_mid_p4, (200,200)))
	count+=1
	if count > 5:
		break
	#print(filename, img.shape, x1, x2, y1, y2, cp, corner1, corner2, corner3, corner4, mid_p1, mid_p2, mid_p3, mid_p4, "\n") #, bbox[1]-bbox[0], bbox[3]-bbox[2])
