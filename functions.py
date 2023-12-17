import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import numpy as np

import matplotlib.pyplot as plt
import cv2

from glob import glob
import os

from tensorflow.data.experimental import AUTOTUNE

IMG_WIDTH = 128
IMG_HEIGHT = 128
BUFFER_SIZE = 1000
BATCH_SIZE = 1

def preprocess_training_dataeset(dataset):
    '''
    To preprocesses the training dataset, we map each image in the dataset
    to its result from preprocess_training_image(), then return the resulting dataset
    '''
    dataset = dataset.cache().map(
        preprocess_training_image,
        num_parallel_calls=AUTOTUNE).shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

    return dataset

def preprocess_training_image(image, label):
    '''
    To preprocess a training image, we apply random_jitter(), which was recommended in this
    article: https://python.plainenglish.io/cycle-gans-generating-paintings-of-my-campus-in-van-goghs-style-9d578b11edf6

    Then, we normalize the image.
    '''
    image = random_jitter(image)
    image = normalize(image, label)
    return (image, label)

def preprocess_testing_dataset(dataset):
    '''
    For preprocessing the testing images, we just normalize them
    '''
    dataset = dataset.cache().map(
        normalize,
        num_parallel_calls=AUTOTUNE).shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

    return dataset

def dataset_to_numpy(dataset):
    """
    This function converts the tensorflow dataset into a numpy array
    It also resizes the images from 256x256 to IMG_WIDTHxIMG_HEIGHT
    """
    images = []
    labels = []

    # Iterate over a dataset
    # for i, (image,label) in enumerate(tfds.as_numpy(dataset)):
    for i, (image, label) in enumerate(tfds.as_numpy(dataset)):
        # BUG FIX: for some reason the preprocessing changes the shape of the images from
        # (IMG_WIDTH, IMG_HEIGHT, 3) to (1, IMG_WIDTH, IMG_HEIGHT)
        # I have no idea why
        image = image[0]

        image = cv2.resize(image,(IMG_WIDTH,IMG_HEIGHT))
        images.append(image)
        # labels.append(label)
        
    return images, labels

def random_crop(image):
    '''
    Randomly crop the image
    '''
    cropped_image = tf.image.random_crop(
    image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

def normalize(image, label=""):
    '''
    This method normalizes the images to be between [-1, 1]
    '''
    image = tf.cast(image, tf.float32)

    normalize_divisor = float(IMG_HEIGHT) - 0.5
    image = (image / normalize_divisor) - 1
    # image = (image / 127.5) - 1
    return image

def random_jitter(image):
    '''
    Applies random jitter according to
    https://python.plainenglish.io/cycle-gans-generating-paintings-of-my-campus-in-van-goghs-style-9d578b11edf6
    '''
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    return image
    
def plot_images(style_images, normal_images):
    '''
    This method selects a few random images from the datasets and plots them
    '''
    # plotting the van goph paintings
    plt.figure(figsize=(7,7))
    plt.title("Van Gogh Images")
    for i,image in enumerate(style_images.shuffle(10000).take(16)):
        plt.subplot(4,4,i+1)
        plt.imshow(image[0])
        plt.axis("off")
    plt.show()

    # plotting the "real world images"
    plt.figure(figsize=(7,7))
    plt.title("Real World Images")
    for i,image in enumerate(normal_images.shuffle(10000).take(16)):
        plt.subplot(4,4,i+1)
        plt.imshow(image[0])
        plt.axis("off")
    plt.show()

def downsample(filters,size,apply_instancenorm=True):
    '''
    TODO: add explanation
    '''
    initializer = tf.random_normal_initializer(0.,0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters,size,strides=2,padding="same",kernel_initializer=initializer,use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result

def upsample(filters,size,apply_dropout=False):
    '''
    TODO: add explanation
    '''
    initializer = tf.random_normal_initializer(0.,0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters,size,strides=2,padding="same",kernel_initializer=initializer,use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))


    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())
    return result


# we need to downsample and upsample the images, so let's write two new layers, Upsample layer and Downsample Layer
OUTPUT_CHANNELS = len(["Red","Green","Blue"])

def Generator():
    inputs = layers.Input([IMG_WIDTH,IMG_HEIGHT,3])

    down_stack = [downsample(128,4), # 64x64x128
                  downsample(256,4), # 32x32x256
                  downsample(512,4), # 16x16x512
                  downsample(512,4), # 8x8x512
                  downsample(512,4), # 4x4x512
                  downsample(512,4), # 2x2x512
                  downsample(512,4), # 1x1x512
                 ]

    up_stack = [upsample(512,4,apply_dropout=True), # 2x2
                upsample(512,4,apply_dropout=True), # 4x4
                upsample(512,4), # 8x8
                upsample(256,4), # 16x16
                upsample(128,4), # 32x32
                upsample(64,4),  # 64x64
               ]

    initializer = tf.random_normal_initializer(0.,0.02)
    last = last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')

    # we'll create skip connections like a residual network
    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up,skip in zip(up_stack,skips):
        x = up(x)
        x = layers.Concatenate()([x,skip])

    x = last(x)

    return keras.Model(inputs=inputs,outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0.,0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02)

    inp = layers.Input([IMG_WIDTH,IMG_HEIGHT,3],name="input_image")

    x = inp

    down1 = downsample(64,4,False)(x) # 64x64x64
    down2 = downsample(128,4,False)(down1) # 32x32x128

    zero_pad1 = layers.ZeroPadding2D()(down2)

    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)

    last = layers.Conv2D(1,4,strides=1,kernel_initializer=initializer)(zero_pad2)

    return keras.Model(inputs=inp,outputs=last)

def get_images_from_local_folder():
    PATH = "our_images/*"
    filelist = glob(PATH)
    images = []

    for filename in filelist:
        print(filename)
        img = cv2.imread(filename)
        img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images.append(img)

    return images
