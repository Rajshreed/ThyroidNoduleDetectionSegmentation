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
def crop_combine_and_save(img, msk, corner1, crop_size, save_path, combine_yes_no): # crop_size = (200,200)
	if combine_yes_no:
		crop_img = get_cropped_image(img, corner1, crop_size)
		crop_msk = get_cropped_image(msk, corner1, crop_size)
		img_msk = np.zeros((200,200,3), np.uint8)
		img_msk[:,:,0] = crop_img
		img_msk[:,:,1] = crop_msk
		cv2.imwrite(save_path, img_msk)
	else:
		cv2.imwrite(save_path, get_cropped_image(img, corner1, crop_size))

def create_cropped_images(input_filename, input_image, input_mask, base_save_path): #filename is only png name without path
	png_name = input_filename
	img = input_image
	msk = input_mask
	pnts = np.where(input_mask>0)# rows, cols
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
	save_path = base_save_path+png_name.split('.')[-2]+"_corner1"+".PNG"
	crop_combine_and_save(img, msk, corner1, (200,200), True)

	save_path = base_save_path+png_name.split('.')[-2]+"_corner2"+".PNG"
        crop_combine_and_save(img, msk, corner2, (200,200), True)

#	cv2.imwrite(save_path, get_cropped_image(img, corner2, (200,200)))

	save_path = base_save_path+png_name.split('.')[-2]+"_corner3"+".PNG"
        crop_combine_and_save(img, msk, corner3, (200,200), True)

	save_path = base_save_path+png_name.split('.')[-2]+"corner4"+".PNG"
        crop_combine_and_save(img, msk, corner4, (200,200), True)

	save_path = base_save_path+png_name.split('.')[-2]+"_mid_p1"+".PNG"
        crop_combine_and_save(img, msk, mid_p1, (200,200), True)

	save_path = base_save_path+png_name.split('.')[-2]+"_mid_p2"+".PNG"
        crop_combine_and_save(img, msk, mid_p2, (200,200), True)

	save_path = base_save_path+png_name.split('.')[-2]+"_mid_p3"+".PNG"
        crop_combine_and_save(img, msk, mid_p3, (200,200), True)

	save_path = base_save_path+png_name.split('.')[-2]+"_mid_p4"+".PNG"
        crop_combine_and_save(img, msk, mid_p4, (200,200), True)

	save_path = base_save_path+png_name.split('.')[-2]+"_cp"+".PNG"
        crop_combine_and_save(img, msk, cp, (200,200), True)

#	if nodule_width > 400 or nodule_height > 400 :
	#crop additional 4 sub-images for large nodules
#		save_path = base_save_path+png_name.split('.')[-2]+"_extra_mid_p1"+".PNG"
#		cv2.imwrite(save_path, get_cropped_image(img, extra_mid_p1, (200,200)))

#		save_path = base_save_path+png_name.split('.')[-2]+"_extra_mid_p2"+".PNG"
#		cv2.imwrite(save_path, get_cropped_image(img, extra_mid_p2, (200,200)))

#		save_path = base_save_path+png_name.split('.')[-2]+"_extra_mid_p3"+".PNG"
#		cv2.imwrite(save_path, get_cropped_image(img, extra_mid_p3, (200,200)))

#		save_path = base_save_path+png_name.split('.')[-2]+"_extra_mid_p4"+".PNG"
#		cv2.imwrite(save_path, get_cropped_image(img, extra_mid_p4, (200,200)))
	#print(filename, img.shape, x1, x2, y1, y2, cp, corner1, corner2, corner3, corner4, mid_p1, mid_p2, mid_p3, mid_p4, "\n") #, bbox[1]-bbox[0], bbox[3]-bbox[2])
