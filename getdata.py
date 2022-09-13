import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def get_data():
    images = []
    for img in os.listdir('images/'):
        images.append(plt.imread('images/'+img))

    images_resized = [tf.image.resize(images[i], [256,256]) for i in range(len(images))]

    images_rescaled = tf.cast(images_resized, tf.float64)/255

    masks = []
    for img in os.listdir('masks/'):
        masks.append(plt.imread('masks/'+img))

    masks_resized = [tf.image.resize(masks[i][..., tf.newaxis], [256,256]) for i in range(len(masks))]

    masks_rescaled = tf.cast(masks_resized, tf.float64)

    return images_rescaled, masks_rescaled