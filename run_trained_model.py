import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from functions import Generator, Discriminator, dataset_to_numpy, get_images_from_local_folder, preprocess_testing_dataset
from cycleGAN import CycleGAN

import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    vangogh_generator = Generator() # generates van gogh paintings using natural images
    photo_generator = Generator() # generates natural images using van gogh paintings

    vangogh_discriminator = Discriminator() # determines whether van gogh painting is real or generated
    photo_discriminator = Discriminator() # determines whether natural image is real or generated

    # Initialize the model
    model = CycleGAN(vangogh_generator=vangogh_generator,
                    photo_generator=photo_generator,
                    vangogh_discriminator=vangogh_discriminator,
                    photo_discriminator=photo_discriminator
                    )

    # Load the saved model weights
    model.load_model_weights()

    # Load and proprocess our images
    our_images = get_images_from_local_folder()
    our_images = np.array(our_images,dtype=np.float32)
    our_images = our_images / 127.5 - 1
    our_images = tf.data.Dataset.from_tensor_slices(our_images).batch(1)

    # Generate & plot the results
    plt.figure(figsize=(10,10))
    for i,image in enumerate(our_images):
        prediction = vangogh_generator(image,training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        plt.imshow(X = prediction)
        plt.axis("off")
        plt.show()