'''
File: data_loaders.py
Description:
	Data loader for thyroid ultrasound data.
'''

import numpy
import numpy as np
import os
import glob
import cv2
from keras.utils import Sequence


#Added for ROI segmentation # takes two channels 
class Generator_Thyroid_clickpt_roi_segmentation(Sequence):
	'''
	This class generates data to feed a Thyroid nodule segmentation Unet model.
		'''
	def __init__(self, filenames, labels_folder,  new_size=(512, 512), labels_dict={}, batch_size=1):
		'''
		Description:
			This object loads images to be fed to the neural network. It assumes that
			that the filenames of the image and mask are identical, but that they are located
			at differnent folder.
		Arguments:
			filenames: A list-type that contains the entire path to the location of the image
			labels_folder: path to the folder that contains the labels
			new_size: New size of the images
			labels_dict: A dictionary that indicates how to transform the labels.
		'''
		self.filenames = filenames
		self.new_size = new_size
		self.num_instances = filenames.shape[0]
		self.labels_dict = labels_dict
		self.order = None
		self.labels_folder = labels_folder
		self.c_index = None
		self.batch_size = batch_size


	def preprocess_data(self, original_image, labels_image=None):
		'''
		Description:
			This function takes a jpg image and formats it to serve as an input to the
			network.
		Arguments:
			original_image; A numpy array of height x width x channels
		'''

		# Resize the image
		resized_image = cv2.resize(original_image, self.new_size)
		if (labels_image is None):
			resized_labels = None
		else:
			resized_labels = cv2.resize(np.uint8(labels_image), self.new_size)		
		return resized_image, resized_labels

	def load_data(self, index_to_load):
		'''
		Description:
			This function will load the file in the index 'index_to_load' as well as its mask.
		Arguments:
			index_to_load: Numpy array containing the indexs to be loaded.
		'''
		X_list = list()
		Y_list = list()

		# Check which files must be loaded
		files_to_load = self.filenames[index_to_load]
		# Make sure that files_to_load is a numpy array.
		if type(files_to_load) is np.ndarray:
			pass
		else:
			files_to_load = np.array([files_to_load])

		# Check the number of files to load
		num_files_to_load = files_to_load.shape[0]
		for file in files_to_load:
			# Get the name of the image to load
			folder_parts = file.split('/')
			image_filename = folder_parts[-1]

			# Load the image and the mask
			img = cv2.imread(file)
			if os.path.isfile(self.labels_folder + image_filename):
				lbl = cv2.imread(self.labels_folder + image_filename)
			else:
				lbl = np.zeros(img.shape)

			# Take first two channels. Also, make the mask a binary thing
			img = img[:,:,:2]
			lbl = lbl[:,:,0] > 0
			# Check if the image and labels have the same dimensions
			if np.array_equal(lbl.shape[:2], img.shape[:2]):
				X, Y = self.preprocess_data(img, lbl)
				X_list.append(X)
				Y_list.append(Y)
		return (np.array(X_list), np.array(Y_list))


	def preprocess_single_instace(self, X):
		'''
		This function is identical to load_data,except that now the original image is given
		'''
		X_list, Y_list = self.preprocess_data(X)

		return np.array(X_list)

	def check_if_shuffle(self):
		if self.c_index >= self.num_instances:
			self.order = np.random.permutation(self.num_instances)
			self.c_index = 0

	def generate_data(self):
		'''
		Function used for training a Keras model.
		'''
		while True:
			if self.c_index == None:
				# The first time that the generator is called it will shuffle the instances.
				self.order = np.random.permutation(self.num_instances)
				# It then will load a single image, it will save that this is the first slice,
				# of the first volume. It will also store the number of slices in this volume
				self.X, self.Y = self.load_data(np.array([self.order[0]]))

				self.c_index = 0
			
			else:
				# If there is a volume already loaded, it will give the next slice. If it is done
				# with all the slices, it will load the next volume
				self.c_index += 1
				self.check_if_shuffle()
				self.X, self.Y = self.load_data(np.array([self.order[self.c_index]]))
			print(self.X.shape,self.Y.shape)

			yield self.X[0:self.batch_size,:,:,:], self.Y[0:self.batch_size,:,:]



def main():
	return -1

if __name__ == '__main__':
	# Do nothing
	main()
