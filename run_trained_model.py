import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import cv2
import os

from functions import Generator, Discriminator, dataset_to_numpy, get_images_from_local_folder, preprocess_testing_dataset
from cycleGAN import CycleGAN

import os
import matplotlib.pyplot as plt

vangogh_generator = Generator() # generates van gogh paintings using natural images
photo_generator = Generator() # generates natural images using van gogh paintings

vangogh_discriminator = Discriminator() # determines whether van gogh painting is real or generated
photo_discriminator = Discriminator() # determines whether natural image is real or generated

# initialize the model
model = CycleGAN(vangogh_generator=vangogh_generator,
                photo_generator=photo_generator,
                vangogh_discriminator=vangogh_discriminator,
                photo_discriminator=photo_discriminator
                )

# Load the saved model weights
model.load_model_weights()

our_images = get_images_from_local_folder()
our_images = np.array(our_images,dtype=np.float32)
our_images = our_images / 127.5 - 1
our_images = tf.data.Dataset.from_tensor_slices(our_images).batch(1)

plt.figure(figsize=(10,10))
for i,image in enumerate(our_images.shuffle(10000).take(1)):
    prediction = vangogh_generator(image,training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    plt.imshow(X = prediction)
    plt.axis("off")
plt.show()
# our_images = preprocess_testing_dataset(our_images)

# images = []
# for i, image in enumerate(tfds.as_numpy(our_images)):
#     # BUG FIX: for some reason the preprocessing changes the shape of the images from
#     # (IMG_WIDTH, IMG_HEIGHT, 3) to (1, IMG_WIDTH, IMG_HEIGHT)
#     # I have no idea why
#     image = image[0]

#     image = cv2.resize(image,(128,128))
#     images.append(image)
#     # labels.append(label)

# # print(our_images)
# # print(our_images)
# # our_images, labels = dataset_to_numpy(our_images)
# our_images = tf.data.Dataset.from_tensor_slices(our_images).batch(1)
# # # convert to float32
# # our_images = np.array(our_images,dtype=np.float32)
# # # scale between -1 and 1
# # our_images = our_images / 127.5 - 1
# # # batch
# # our_images = tf.data.Dataset.from_tensor_slices(our_images).batch(1)

# # np.array(our_images,dtype=np.float32)

# for pic in images:
#     plt.figure(figsize=(10,10))
#     prediction = vangogh_generator(pic,training=False)[0].numpy()
#     prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
#     plt.imshow(X = prediction)
#     plt.axis("off")
#     plt.show()

# Loading the van gogh dataset from tensorflow
dataset, metadata = tfds.load('cycle_gan/vangogh2photo',with_info=True, as_supervised=True)
test_van_gogh, test_photo = dataset['testA'], dataset['testB']

test_photo = preprocess_testing_dataset(test_photo)
test_photo, labels = dataset_to_numpy(test_photo)
test_photo = tf.data.Dataset.from_tensor_slices(test_photo).batch(1)

plt.figure(figsize=(10,10))
for i,image in enumerate(test_photo.shuffle(10000).take(36)):
    # plt.subplot(6,6,i+1)
    # plt.imshow(X = image)

    plt.subplot(6,6,i+1)
    prediction = vangogh_generator(image,training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    plt.imshow(X = prediction)
    plt.axis("off")
    if i==36:
        break
plt.show()