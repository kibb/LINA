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
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers

# Define paths for the test set and the pre-trained model
trainingSetPath = 'Training Set' #Please modify the path if needed
testSetPath = 'Test Set' #Please modify the path if needed
modelPath = 'Model' #Please modify the path if needed
savePath = 'Saved Models'

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

# Load training images for Fluorescence
image_list = []
listOfImages=glob.glob(trainingSetPath + '/Fluorescence/*.tif')
imagesLength=len(listOfImages)
for ii in tqdm(range(0,imagesLength)):
    im=plt.imread(listOfImages[ii], 'tif')
    im_max = np.max(im) #for normalization
    im_min = np.min(im) #for normalization
    if(im_max != 0):
        im = (im - im_min)/(im_max - im_min)
    image_list.append(im)
labels = np.array(image_list)
image_list = None

# Load training images for Phase
image_list = []
listOfImages=glob.glob(trainingSetPath + '/Phase/*.tif')
imagesLength=len(listOfImages)
for ii in tqdm(range(0,imagesLength)):
    im = io.imread(listOfImages[ii])
    for jj in range(0,8):
        im_max = np.max(im[jj]) #for normalization
        im_min = np.min(im[jj]) #for normalization
        im[jj] = (im[jj] - im_min)/(im_max - im_min)
    image_list.append(im)
inputImages = np.array(image_list)
inputImages = np.swapaxes(inputImages, 1, 2)
inputImages = np.swapaxes(inputImages, 2, 3)
image_list = None

# Divide the dataset into a test set + trainining/validation set
nOfImages=len(labels)
testSize=int(0.9*nOfImages)

train_images=inputImages[0:testSize]
train_labels=labels[0:testSize]

test_images=inputImages[testSize:]
test_labels=labels[testSize:]

# Reshape training and test images
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], train_images.shape[3], 1)
train_labels = train_labels.reshape(train_labels.shape[0], train_labels.shape[1], train_labels.shape[2], 1)
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], test_images.shape[3], 1)
test_labels = test_labels.reshape(test_labels.shape[0], test_labels.shape[1], test_labels.shape[2], 1)

# Load the pre-trained model
pretrained_model = keras.models.load_model(modelPath + '/PixelRegressionModel.h5')

# Define the ratio of validation data to training data for the new dataset
validtrain_split_ratio = 0.2  # % of the seen dataset to be put aside for validation, rest is for training

# Specify whether to shuffle the training data before each epoch
batch_shuffle = True   # shuffle the training data prior to batching before each epoch

# Learning rate for the optimizer
lrate = 1e-4

# Number of training epochs
epochs = 500

# Batch size for training
batch_size = 10

# Define callback functions
my_callbacks = [
    # Early stopping to prevent overfitting
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=30, verbose=1, mode='auto'),
    
    # Save the best model checkpoints during training
    tf.keras.callbacks.ModelCheckpoint(filepath= savePath + '/model_{epoch:02d}_{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto'),
]

# Create a new output layer for the new task
new_output_layer = Conv2D(1, 1, activation='linear')(pretrained_model.layers[-2].output)

# Create the new model with the pre-trained layers and the new output layer
transfer_model = keras.Model(inputs=pretrained_model.input, outputs=new_output_layer)

# Compile the transfer model with a new loss function and optimizer
transfer_model.compile(
    optimizer=optimizers.Adam(lr=lrate),  # You can adjust the learning rate
    loss='mean_squared_error',  # Use an appropriate loss function for your task
    metrics=['mean_absolute_error']  # Add relevant metrics
)

# Train the transfer model on the new dataset
transfer_history = transfer_model.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    callbacks = my_callbacks ,
    validation_split=validtrain_split_ratio,
    shuffle=batch_shuffle,
    verbose=1
)

# After training, you can use the transfer_model for predictions on your new images
new_test_predict = transfer_model.predict(test_images)

# plot the model loss

plt.plot(transfer_history.transfer_history['loss'])
plt.plot(transfer_history.transfer_history['val_loss'])
plt.ylabel('Loss [MSE]')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='best')
plt.show()

# plot the model accuracy metric
metrics = ['mean_absolute_error']
plt.plot(np.array(transfer_history.transfer_history[metrics[0]]))
plt.plot(np.array(transfer_history.transfer_history['val_' + metrics[0]]))
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='best')
plt.show()