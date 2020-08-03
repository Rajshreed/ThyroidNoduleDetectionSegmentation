'''
Description:
	This file contain some custom loss functions to be used with the models
'''
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coe(output, target, smooth=1):
	target = tf.cast(target, tf.float32)
	output = tf.cast(output, tf.float32)
	intersection = tf.reduce_sum(output * target)
	dice = (2. * intersection + smooth) / (tf.reduce_sum(output) + tf.reduce_sum(target) + smooth)
	return (1 - dice)

def dice_coeff_with_logits(target, logits, smooth=1, weights=None):
	'''
	This function operates with tensors
	'''
	target = tf.cast(target, tf.float32)
	y_hat = tf.nn.softmax(logits)
	output = tf.cast(y_hat[:,1], tf.float32)
	intersection = tf.reduce_sum(output * target)
	dice = (2. * intersection + smooth) / (tf.reduce_sum(output) + tf.reduce_sum(target) + smooth)
	return (1 - dice)

def sigmoid_dice_coeff_with_logits(target, logits, smooth=1, weights=None):
	'''
	This function operates with tensors
	'''
	target = tf.cast(target, tf.float32)
	y_hat = tf.nn.sigmoid(logits)
	output = tf.cast(y_hat, tf.float32)
	intersection = tf.reduce_sum(output * target)
	dice = (2. * intersection + 1) / (tf.reduce_sum(output) + tf.reduce_sum(target) + 1)
	return (1 - dice)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred):
    SMOOTH = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_and_balanced_cross_entropy_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred) + balanced_cross_entropy(y_true, y_pred)

def centroid_distance_old(y_true, y_pred):
    y_true_f = tf.squeeze(y_true)
    y_pred_f = tf.squeeze(y_pred)

    indices_true = tf.cast( tf.where(tf.greater(y_true_f,0.0)), tf.float32)
    indices_pred = tf.cast( tf.where(tf.greater(y_pred_f,0.0)), tf.float32)

    true_mean = tf.cast(tf.reduce_mean(indices_true, 0), tf.float32 ) 
    pred_mean = tf.cast(tf.reduce_mean(indices_pred, 0), tf.float32 )  

    distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(true_mean, pred_mean))))
    distance1 = tf.cond( tf.less(distance, 10.0), lambda:tf.constant(0.0), lambda: tf.cast(distance,tf.float32) )  # 10 is the grace distance or tolerance 
    return distance1


def centroid_distance(y_true, y_pred):
    y_true_f = tf.squeeze(y_true)
    y_pred_f = tf.squeeze(y_pred)

    indices_true = tf.cast( tf.where(y_true_f >= 0), tf.float32)
    indices_pred = tf.cast( tf.where(y_pred_f >= 0), tf.float32)

    indices_true = tf.reshape(indices_true, [512,512,2])
    indices_pred = tf.reshape(indices_pred, [512,512,2])

    weighted_avg_indices_true = tf.divide(tf.multiply(y_true_f, indices_true), tf.reduce_sum(y_true_f))
    weighted_avg_indices_pred = tf.divide(tf.multiply(y_pred_f, indices_pred), tf.reduce_sum(y_pred_f))

    true_mean = tf.cast(tf.reduce_mean(weighted_avg_indices_true, 0), tf.float32 ) 
    pred_mean = tf.cast(tf.reduce_mean(weighted_avg_indices_pred, 0), tf.float32 )  

    distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(true_mean, pred_mean))))
    distance1 = tf.cond( tf.less(distance, 10.0), lambda:tf.constant(0.0), lambda: tf.cast(distance,tf.float32) )  # 10 is the grace distance or tolerance 

def balanced_cross_entropy(y_true, y_pred, beta=0.9):
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))
    def loss(y_true, y_pred, beta):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss(y_true, y_pred, beta)


def loss_new(y_true, y_pred):
    def dice_loss_new(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))
    return binary_crossentropy(y_true, y_pred) + dice_loss_new(y_true, y_pred)


def weighted_sigmoid_cross_entropy_with_logits(y_true, y_pred):
	weight_pos = 4

	return tf.nn.weighted_cross_entropy_with_logits(
		targets=y_true, logits=y_pred, pos_weight=weight_pos)

def dice_coef_border(y_true, y_pred):
    SMOOTH=1
    print("inside dice_coef_border .. y_true shape=", y_true.shape, "y_pred shape=", y_pred.shape)

#    negative = 1. - tf.cast(y_true, tf.float32) 
#    positive = tf.cast(y_true, tf.float32) 

    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number

    averaged_mask1 = tf.nn.avg_pool2d( y_true, pool_size=(11, 11), strides=(1, 1), padding='same')#, pool_mode='avg')
    border1 = K.cast(K.greater(averaged_mask1, 0.005), 'float32') * K.cast(K.less(averaged_mask1, 0.995), 'float32')


#    positive = K.pool2d(positive, pool_size=(11,11), padding="same", data_format='channels_last')
#    negative = K.pool2d(negative, pool_size=(11,11), padding="same", data_format='channels_last')
#    border = positive * negative

#    border = get_border_mask((11, 11), y_true)


    averaged_mask2 = tf.nn.avg_pool2d( y_pred, pool_size=(11, 11), strides=(1, 1), padding='same')#, pool_mode='avg')
    border2 = K.cast(K.greater(averaged_mask2, 0.005), 'float32') * K.cast(K.less(averaged_mask2, 0.995), 'float32')

    border1 = K.flatten(border1)
    border2 = K.flatten(border2)

    y_true_f = border1#K.flatten(y_true)
    y_pred_f = border2#K.flatten(y_pred)
#    y_true_f = K.gather(y_true_f, tf.where(border1 > 0.5))
#    y_pred_f = K.gather(y_pred_f, tf.where(border2 > 0.5))
    
    print("y_pred_f shape",y_pred_f.shape, "y_true_f.shape=", y_true_f.shape)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)
#    return K.sum(border)

#    return dice_coef(y_true_f, y_pred_f)
#    return dice_coef(y_true, y_pred)

def get_border_mask(pool_size, y_true):
    y_true = tf.cast(y_true, tf.float32) 
    print("y_true.shape=",y_true.shape)
    negative = 1 - y_true
    positive = y_true
    print("positive.shape= ",positive.shape)
    print("negative.shape= ",negative.shape)

#    positive = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(positive)
#    negative = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(negative)
#    positive = tf.squeeze(positive)
#    negative = tf.squeeze(negative)

#    positive = tf.nn.avg_pool2d(positive, pool_size=pool_size, padding="same")
#    negative = tf.nn.avg_pool2d(negative, pool_size=pool_size, padding="same")

#    positive = K.pool2d(positive, pool_size=pool_size, padding="same", data_format='channels_last')
#    negative = K.pool2d(negative, pool_size=pool_size, padding="same", data_format='channels_last')
#    positive = tf.expand_dims(positive,axis=0)
#    negative = tf.expand_dims(negative,axis=0)

    border = tf.uint8(positive * negative)
    print("after-positive.shape= ",positive.shape)
    print("after negative.shape= ",negative.shape)
    print("border.shape= ",border.shape)
#    return border
    return border

def dice_coef_loss_border(y_true, y_pred):
    return (- dice_coef_border(y_true, y_pred)) * 0.50 + 0.50 * (-dice_coef(y_true, y_pred))

def dice_coef_and_center_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred) + distance_loss_to_center(y_true, y_pred)

def distance_loss_to_center(labels, logits):
    image_size = 512
    ZERO_DIV_OFFSET = 0.00001

    # Make array of coordinates (each row contains three coordinates)
    ii, jj = tf.meshgrid(
        tf.range(image_size), tf.range(image_size), indexing='ij')
    coords = tf.stack([tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,))], axis=-1)
    coords = tf.cast(coords, tf.int32)
    coords = tf.cast(coords, tf.float32)

    # Rearrange input into one vector per volume
    volumes_flat = tf.reshape(labels,
                              [-1, image_size * image_size * 1, 1])

    # Compute total mass for each volume. Add 0.00001 to prevent division by 0
    total_mass = tf.cast(tf.reduce_sum(volumes_flat, axis=1),
                         tf.float32) + ZERO_DIV_OFFSET

    # Compute centre of mass
    center = tf.cast(tf.reduce_sum(volumes_flat * coords, axis=1),
                     tf.float32) / total_mass
    center = center / image_size

    # Normalize coordinates by size of image
    logits = logits / image_size

    distance_to_center = tf.sqrt(
        tf.reduce_sum(tf.square(logits - center), axis=-1) + ZERO_DIV_OFFSET)
    return distance_to_center

#    return tf.losses.mean_squared_error(center, logits)
