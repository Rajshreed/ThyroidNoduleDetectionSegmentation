# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
from glob import glob
import cv2
 
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to the input image")
#ap.add_argument("-o", "--output", required=True,#
#	help="path to output directory to store augmentation examples")
#ap.add_argument("-t", "--total", type=int, default=100,
#	help="# of training samples to generate")
#args = vars(ap.parse_args())


# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image...")
#image = load_img(args["image"])
#image = img_to_array(image)
#image = np.expand_dims(image, axis=0)
 
# construct the image generator for data augmentation then
# initialize the total number of images generated thus far
aug = ImageDataGenerator(
        rotation_range=15,
        zoom_range=1,#0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        #vertical_flip=True,
        horizontal_flip=True,
        brightness_range=(.5,1),
        fill_mode="constant", cval=0)
total = 0

# construct the actual Python generator
print("[INFO] generating images...")
#for filename in glob("./DataAug/Balanced_merged/*.png"):#("./DataAug/OriginalData/*.png"):
for filename in glob("./img_msk/*.PNG"):#("./DataAug/OriginalData/*.png"):
    total = 0
    image = load_img(filename)
    image = img_to_array(image)
    mask_border_sum = np.sum(image[0:1,:,1]) + np.sum(image[:,0:1,1]) + np.sum( image[499:500, :, 1]) + np.sum(image[:, 699:700, 1])
    if mask_border_sum != 0:
        continue
    image = np.expand_dims(image, axis=0)
    save_prefix=filename.split('/')[-1].split('.')[-2]
    save_to_dir='./img_msk_aug/'
    #cv2.imshow("show",np.uint8(image))
    #cv2.waitKey(5000)
    imageGen = aug.flow(image, batch_size=1 )
#   imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],save_prefix="image", save_format="png")
 
# loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        mask_border_sum = np.sum(image[0,0:1,:,1]) + np.sum(image[0,:,0:1,1]) + np.sum( image[0,499:500, :, 1]) + np.sum(image[0,:, 699:700, 1])
        if (mask_border_sum == 0 and np.sum(image[0,:,:,1])>0):
            total += 1
            dir_save = save_to_dir + save_prefix + "_aug" + str(total) + ".png" 
            cv2.imwrite(dir_save,image[0,:,:,:]) 
           # if we have reached the specified number of examples, break
        # from the loop
        if total == 25:#100:
            break

