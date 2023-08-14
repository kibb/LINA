#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from PIL import Image
from skimage import io

from tensorflow.keras import layers 
from tensorflow.keras.layers import Input , Conv2D , MaxPooling2D , Dropout , concatenate , UpSampling2D, MaxPooling3D, Conv3D, Reshape, Activation

# Define paths for the test set and the pre-trained model
testSetPath = 'Test Set' #Please modify the path if needed
modelPath = 'Model' #Please modify the path if needed


# Define a function to create a U-Net model
def UNet(input_shape):
    keras.backend.clear_session()
    
    # Input
    input_layer = Input(input_shape)
    nb_kernels = 16

    # Downsampling path
    conv1 = Conv3D(nb_kernels, (3, 3, 8), activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
    conv1 = Conv3D(nb_kernels, (3, 3, 8), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 8))(conv1)
    pool1_reshaped = Reshape((176, 176, nb_kernels))(pool1)
    conv1_upsampled = Reshape((352, 352, nb_kernels * 8))(conv1)

    conv2 = Conv2D(nb_kernels * 2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1_reshaped)
    conv2 = Conv2D(nb_kernels * 2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_kernels * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(nb_kernels * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_kernels * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(nb_kernels * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Upsampling path
    conv5 = Conv2D(nb_kernels * 16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(nb_kernels * 16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(nb_kernels * 8, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(nb_kernels * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(nb_kernels * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(nb_kernels * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(nb_kernels * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(nb_kernels * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(nb_kernels * 2, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(nb_kernels * 2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(nb_kernels * 2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(nb_kernels, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1_upsampled, up9], axis=3)
    conv9 = Conv2D(nb_kernels, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(nb_kernels, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # Output layer
    outputs = Conv2D(1, 1, activation=tf.keras.activations.linear)(conv9)

    # Create and compile the U-Net model
    model = keras.Model(inputs=input_layer, outputs=outputs, name='UNet')

    return model

# Load test images for Fluorescence
image_list = []
listOfImages=glob.glob(testSetPath + '/Fluorescence/*.tif')
imagesLength=len(listOfImages)
for ii in tqdm(range(0,imagesLength)):
    im=plt.imread(listOfImages[ii], 'tif')
    im_max = np.max(im) #for normalization
    im_min = np.min(im) #for normalization
    if(im_max != 0):
        im = (im - im_min)/(im_max - im_min)
    image_list.append(im)
test_labels = np.array(image_list)
image_list = None

# Load test images for Phase
image_list = []
listOfImages=glob.glob(testSetPath + '/Phase/*.tif')
imagesLength=len(listOfImages)
for ii in tqdm(range(0,imagesLength)):
    im = io.imread(listOfImages[ii])
    for jj in range(0,8):
        im_max = np.max(im[jj]) #for normalization
        im_min = np.min(im[jj]) #for normalization
        im[jj] = (im[jj] - im_min)/(im_max - im_min)
    image_list.append(im)
test_images = np.array(image_list)
test_images = np.swapaxes(test_images, 1, 2)
test_images = np.swapaxes(test_images, 2, 3)
image_list = None

# Reshape test images
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], test_images.shape[3], 1)
test_labels = test_labels.reshape(test_labels.shape[0], test_labels.shape[1], test_labels.shape[2], 1)

# Load the pre-trained U-Net model
model = keras.models.load_model(modelPath + '/PixelRegressionModel.h5')

# Make predictions using the loaded model on test images
test_predict = model.predict(test_images)