import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate, Input, Conv2D, MaxPooling2D, Reshape, Conv2DTranspose, concatenate, BatchNormalization, Activation
np.random.seed(42)
tf.random.set_seed(42)
from sklearn.model_selection import train_test_split

from keras import backend as K

def unet():
    input = Input([256,256,3])
    x = Conv2D(64, 3, activation='relu', padding='same')(input)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(c1)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(c2)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(256, 3, 2, activation='relu', padding='same')(c3)
    x = concatenate([x, c2])
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, 3, 2, activation='relu', padding='same')(x)
    x = concatenate([x, c1])
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(input, x)

def simple_unet():
    input = Input([256,256,3])
    x = Conv2D(64, 3, activation='relu', padding='same')(input)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(c1)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(c2)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(256, 3, 2, activation='relu', padding='same')(c3)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, 3, 2, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(input, x)

def unet_dropout():
    input = Input([256,256,3])
    x = Conv2D(64, 3, activation='relu', padding='same')(input)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(c1)
    x = Dropout(0.3)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(c2)
    x = Dropout(0.3)(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(256, 3, 2, activation='relu', padding='same')(c3)
    x = concatenate([x, c2])
    x = Dropout(0.3)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, 3, 2, activation='relu', padding='same')(x)
    x = concatenate([x, c1])
    x = Dropout(0.3)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(input, x)

def unet_batchnormalization():

    def Conv2D_block(input, filters):
        x = Conv2D(filters, 3, padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation('relu')(x)
    
    def Conv2DTranspose_block(input, filters):
        x = Conv2DTranspose(filters, 3, 2, padding='same')(input)
        x = BatchNormalization()(x)
        return Activation('relu')(x)

    input = Input([256,256,3])
    c1 = Conv2D_block(input, 64)
    x = MaxPooling2D()(c1)
    c2 = Conv2D_block(x, 128)
    x = MaxPooling2D()(c2)
    c3 = Conv2D_block(x, 256)
    x = Conv2DTranspose_block(c3, 256)
    x = concatenate([x, c2])
    x = Conv2D_block(x, 128)
    x = Conv2DTranspose_block(x, 128)
    x = concatenate([x, c1])
    x = Conv2D_block(x, 64)
    x = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(input, x)

def unet_vgg16(trainable = True):
    b_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

    if trainable == False:
        for layer in b_model.layers:
            layer.trainable = False

    input = Input([256,256,3])
    x = b_model.layers[1](input)
    c1 = b_model.layers[2](x)
    x = b_model.layers[3](c1)
    x = b_model.layers[4](x)
    c2 = b_model.layers[5](x)
    x = b_model.layers[6](c2)
    x = b_model.layers[7](x)
    x = b_model.layers[8](x)
    x = Conv2DTranspose(256, 3, 2, activation='relu', padding='same')(x)
    x = concatenate([x, c2])
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, 3, 2, activation='relu', padding='same')(x)
    x = concatenate([x, c1])
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(input, x)

