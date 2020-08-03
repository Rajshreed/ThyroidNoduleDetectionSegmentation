import numpy as np

# import Keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import model_from_json
# custom packages
from losses import dice_coef_loss, dice_coef  

import tensorflow as tf

''' Create the network model

		Returns:
			model, gpu_model: gpu_model is for multi-gpu training
'''

''' Create the network model '''

def get_unet2(lr=1e-5, img_row=512, img_cols=512, multigpu=1, channels_count=1):
	inputs = Input(shape=(img_row, img_cols, channels_count), name='net_input') # multi channel input
	conv1 = Conv2D(32, (3, 3), data_format='channels_last', activation='relu', padding='same')(inputs)
	conv1 = Conv2D(32, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), data_format='channels_last',activation='relu', padding='same')(pool1)
	conv2 = Conv2D(64, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), data_format='channels_last',activation='relu', padding='same')(pool2)
	conv3 = Conv2D(128, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), data_format='channels_last',activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), data_format='channels_last',activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv5)
	pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

	conv6 = Conv2D(1024, (3, 3), data_format='channels_last',activation='relu',padding='same')(pool5)
	conv6 = Conv2D(1024, (3, 3), data_format='channels_last',activation='relu',padding='same')(conv6)
	pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

	conv7 = Conv2D(2048, (3, 3), data_format='channels_last',activation='relu',padding='same')(pool6)
	conv7 = Conv2D(2048, (3, 3), data_format='channels_last',activation='relu',padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(1024, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv7),conv6], axis=3)
	conv8 = Conv2D(1024, (3, 3), data_format='channels_last',activation='relu', padding='same')(up8)
	conv8 = Conv2D(1024, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(512, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv8),conv5], axis=3)
	conv9 = Conv2D(512, (3, 3), data_format='channels_last',activation='relu', padding='same')(up9)
	conv9 = Conv2D(512, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv9)

	up10 = concatenate([Conv2DTranspose(256, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv9), conv4], axis=3)
	conv10 = Conv2D(256, (3, 3), data_format='channels_last',activation='relu', padding='same')(up10)
	conv10 = Conv2D(256, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv10)

	up11 = concatenate([Conv2DTranspose(128, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv10), conv3], axis=3)
	conv11 = Conv2D(128, (3, 3), data_format='channels_last',activation='relu', padding='same')(up11)
	conv11 = Conv2D(128, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv11)

	up12 = concatenate([Conv2DTranspose(64, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv11), conv2], axis=3)
	conv12 = Conv2D(64, (3, 3), data_format='channels_last',activation='relu', padding='same')(up12)
	conv12 = Conv2D(64, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv12)

	up13 = concatenate([Conv2DTranspose(32, (2, 2), data_format='channels_last',strides=(2, 2), padding='same')(conv12), conv1], axis=3)
	conv13 = Conv2D(32, (3, 3), data_format='channels_last',activation='relu', padding='same')(up13)
	conv13 = Conv2D(32, (3, 3), data_format='channels_last',activation='relu', padding='same')(conv13)

	conv14 = Conv2D(1, (1, 1), data_format='channels_last',activation='sigmoid', name='net_output')(conv13)

	model = Model(inputs=[inputs], outputs=[conv14])

	model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])
	return model

