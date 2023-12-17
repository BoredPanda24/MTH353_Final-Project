import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import os

from functions import *
from cycleGAN import CycleGAN

from tensorflow.data.experimental import AUTOTUNE

if __name__ == "__main__":
    # Loading the van gogh dataset from tensorflow
    dataset, metadata = tfds.load('cycle_gan/vangogh2photo',with_info=True, as_supervised=True)
    train_van_gogh, train_photo = dataset['trainA'], dataset['trainB']
    test_van_gogh, test_photo = dataset['testA'], dataset['testB']

    ############## testing preprocessing ##############

    train_van_gogh = preprocess_training_dataeset(train_van_gogh)
    train_photo = preprocess_training_dataeset(train_photo)

    test_van_gogh = preprocess_testing_dataset(test_van_gogh)
    test_photo = preprocess_testing_dataset(test_photo)
    
    sample_van_gogh = next(iter(train_van_gogh))
    sample_photo = next(iter(train_photo))

    plt.figure()
    plt.imshow(sample_van_gogh[0][0] * 0.5 + 0.5)
    plt.axis(False)
    plt.figure()
    plt.imshow(sample_photo[0][0]*0.5 + 0.5)
    plt.axis(False)
    # plt.show()

    ############## end testing preprocessing ##############

    # convert the tensorflow dataset into numpy arrays
    style_images, labels = dataset_to_numpy(train_van_gogh)
    normal_images, labels = dataset_to_numpy(train_photo)

    # convert to float32
    # style_images = np.array(style_images,dtype=np.float32)
    # normal_images = np.array(normal_images,dtype=np.float32)

    # # scale between -1 and 1
    # style_images = style_images / 127.5 - 1
    # normal_images = normal_images / 127.5 - 1

    # batch
    style_images = tf.data.Dataset.from_tensor_slices(style_images).batch(1)
    normal_images = tf.data.Dataset.from_tensor_slices(normal_images).batch(1)

    # plot the images
    # plot_images(style_images, normal_images)

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
    
    # train the model
    model.train_model(style_images, normal_images)

    # Generating paintings
    plt.figure(figsize=(10,10))
    for i,image in enumerate(normal_images.shuffle(10000).take(36)):
        plt.subplot(6,6,i+1)
        prediction = vangogh_generator(image,training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        plt.imshow(X = prediction)
        plt.axis("off")
        if i==36:
            break
    plt.show()