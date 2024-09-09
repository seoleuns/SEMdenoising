## Based on https://keras.io/examples/generative/ddim/ 
#
## Modified for SEM-Denoising by Seoleun Shin (KRISS, 2023-05-01)
##

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import History
import matplotlib.image as img
import numpy as np
import math
import re
import os
import random
import cv2
from PIL import Image
import imageio
import shutil
import glob
from tensorflow.python.client import device_lib

from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3


device_lib.list_local_devices()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.config.list_physical_devices('GPU')

model_name = 'InceptionV3'

def cropImage(Image, XY: tuple, WH: tuple, returnGrayscale=False):
    # Extract the x,y and w,h values
    (x, y) = XY
    (w, h) = WH
    # Crop Image with numpy splitting
    crop = Image[y:y + h, x:x + w]
    # Check if returnGrayscale variable is true if is then convert image to grayscale
    if returnGrayscale:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Return cropped image
    return crop


def rotateImage(image, angle):
    row,col = image.shape[0:2]
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image



def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)

GCS_PATH = './Data'
fn_photo = tf.io.gfile.glob(str(GCS_PATH + '/time/high/*.tif'))
fn_photo1 = tf.io.gfile.glob(str(GCS_PATH + '/time/hightest/*.tif'))
fn_noise = tf.io.gfile.glob(str(GCS_PATH + '/time/low/*.tif'))


BATCH_SIZE = 4

def parse_function(filename):
    image = cv2.imread(filename.decode('float32'), cv2.IMREAD_UNCHANGED)
    return image

from PIL import Image

def tiff_image_generator(filename):
    for path in filename:
        # Read TIFF image using PIL
        image = Image.open(path).convert('L')
        image = np.array(image)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        
        # Convert image to NumPy array or TensorFlow tensor
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        
        # Perform any necessary preprocessing on the image
        
        yield image


# Perform any additional operations on the dataset


def preprocess_image(data):
    # center crop image
    height = tf.shape(data)[0]
    width = tf.shape(data)[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data,
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)

def data_augment(image):
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    if p_crop > 0.5:
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.image.random_crop(image, size=[image_size, image_size, 3])
        if p_crop > 0.9:
            image = tf.image.resize(image, [image_size, image_size])
            image = tf.image.random_crop(image, size=[image_size, image_size, 3])
    if p_rotate > 0.9:
        image = tf.image.rot90(image, k=3)
    elif p_rotate > 0.7:
        image = tf.image.rot90(image, k=2)
    elif p_rotate > 0.5:
        image = tf.image.rot90(image, k=1)
    if p_spatial > 0.6:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        if p_spatial > 0.9:
            image = tf.image.transpose(image)
    return image
#num_parallel_calls=tf.data.experimental.AUTOTUNE
num_parallel_calls=tf.data.AUTOTUNE

def getSet(filenames):
    # Create a TensorFlow dataset from the custom generator function
    dataset = tf.data.Dataset.from_generator(
        tiff_image_generator,
        args=[filenames],
        output_signature=tf.TensorSpec(shape=(471, 512, 3), dtype=tf.uint8)
    )
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(preprocess_image, num_parallel_calls)
    dataset = dataset.map(data_augment, num_parallel_calls)
    dataset = dataset.repeat(150)
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(num_parallel_calls)
    return dataset

#=========******************
#DIFF
dataset_repetitions = 5
num_epochs = 50  # train for at least 50 epochs for good results
image_size = 512
# KID = Kernel Inception Distance, see related section
kid_image_size = 286
kid_diffusion_steps = 20
plot_diffusion_steps = 20
test_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.97

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128, 256]
block_depth = 2

# optimization
batch_size = 4
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


def prepare_dataset(split):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    return (
        tfds.load(dataset_name, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


# load dataset
train_dataset = getSet(fn_photo)
val_dataset = getSet(fn_photo1)
test_dataset = getSet(fn_noise)


import numpy as np


class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        modelpath = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        inceptionmodel = InceptionV3(include_top=False, weights=modelpath)
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                #layers.Lambda(inceptionmodel.preprocess_input),
                #layers.Lambda(InceptionV3.preprocess_input),
                keras.applications.InceptionV3(
                #inceptionmodel(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
                    #weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def testgenerate(self, batch_images, diffusion_steps):
        # noise -> images -> denormalized images
        testelement = test_dataset.take(batch_images)
        #generated_images = tf.zeros(shape=(batch_size*batch_images, image_size, image_size, 3))
        for element in testelement:
            initial_noise = element[:8]
            generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
            generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}


    def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=4):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=(num_rows) * (num_cols),
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig('default_%s.png'%(epoch))
        plt.close()

    def testplot_images(self, epoch=None, logs=None, num_rows=2, num_cols=8):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.testgenerate(
            batch_images = 1,
            diffusion_steps=test_diffusion_steps,
        )
        testelement = test_dataset.take(1)
        for element in testelement:
            noisy_images = element[:8]

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(noisy_images[col])
                plt.axis("off")
                plt.subplot(num_rows, num_cols, index + 9)
                plt.imshow(generated_images[col])
                plt.axis("off")
        plt.tight_layout()
        plt.savefig('plottest.png')
        plt.show()
        plt.close()


    def call(self, inputs):
        x = self.dense1(inputs)
        outputs = self.dense2(x)
        return outputs



# create and compile the model
model = DiffusionModel(image_size, widths, block_depth)
# below tensorflow 2.9:
# pip install tensorflow_addons
# import tensorflow_addons as tfa
# optimizer=tfa.optimizers.AdamW
model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)
# pixelwise mean absolute error is used as loss

# save the best model based on the validation KID metric
#checkpoint_path = "checkpoints/diffusion_model"
checkpoint_path = "./diffmodelsave"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="max",
    save_best_only=False,
    save_freq='epoch'
)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(train_dataset)

#model.save_weights(checkpoint_path.format(epoch=0))
# run training and plot generated images periodically
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    verbose = 0,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback,
    ],
)

# load the best model and generate images
model.load_weights(checkpoint_path)
model.testplot_images()
exit()

