## For the evaluation of U-Net+Attention
## 
## U-Net + spatial attention mechanism implemented by Seoleun Shin (KRISS) 2024/September/01 ##
##
## Also use metrics based on muller2020super and modified for our samples (Seoleun Shin, 2023/June/01) ##
##---------------------------------------------------------------------------------------------------------
##
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compssim
from image_similarity_measures.evaluate import evaluation
from tensorflow.python.ops.numpy_ops import np_config
from skimage import img_as_float
np_config.enable_numpy_behavior()

device_lib.list_local_devices()

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'


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


import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, LeakyReLU
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
from keras import regularizers
import keras.backend as kb
from tensorflow.keras import layers


target_width = 512
target_height = 512
image_size =512
np.random.seed(456)

def renormal_load_image_from_dir(img_path):
    file_list = glob.glob(img_path+'/*.tif')
    file_list.sort()
    img_list = np.empty((len(file_list), target_height, target_width, 1))
    for i, fig in enumerate(file_list):
        figures = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
        figures = cropImage(figures,(0,0),(512,440))
        img = tf.keras.utils.img_to_array(figures).astype('float32')
        img_array = tf.image.resize(img, size=[image_size, image_size], antialias=True)
        img_list[i] = tf.clip_by_value(img_array/255.0, 0.0, 1.0)
    return img_list

def train_test_split(data,random_seed=55,split=0.75):
    set_rdm = np.random.RandomState(seed=random_seed)
    dsize = len(data)
    ind = set_rdm.choice(dsize,dsize,replace=False)
    train_ind = ind[:int(split*dsize)]
    val_ind = ind[int(split*dsize):]
    return data[train_ind],data[val_ind]

def augment_pipeline(pipeline, images, seed=5):
    ia.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = np.array(step.augment_images(images))
        processed_images = np.append(processed_images, temp, axis=0)
    return(processed_images)


rotate90 = iaa.Rot90(1)          # rotate image 90 degrees
rotate180 = iaa.Rot90(2)         # rotate image 180 degrees
rotate270 = iaa.Rot90(3)         # rotate image 270 degrees
random_rotate = iaa.Rot90((1,3)) # randomly rotate image from 90,180,270 degrees
perc_transform = iaa.PerspectiveTransform(scale=(0.02, 0.1)) # Skews and transform images without black bg
rotate3 = iaa.Affine(rotate=(3))         # rotate image 3 degrees
rotate3r = iaa.Affine(rotate=(-3))       # rotate image 3 degrees in reverse
rotate5 = iaa.Affine(rotate=(5))         # rotate image 5 degrees
rotate5r = iaa.Affine(rotate=(-5))       # rotate image 5 degrees in reverse
rotate7 = iaa.Affine(rotate=(7))         # rotate image 7 degrees
rotate7r = iaa.Affine(rotate=(-7))       # rotate image 7 degrees in reverse
rotate10 = iaa.Affine(rotate=(10))       # rotate image 10 degrees
rotate10r = iaa.Affine(rotate=(-10))     # rotate image 10 degrees in reverse
rotate15 = iaa.Affine(rotate=(15))       # rotate image 15 degrees
rotate15r = iaa.Affine(rotate=(-15))     # rotate image 15 degrees in reverse
rotate20 = iaa.Affine(rotate=(20))       # rotate image 20 degrees
rotate20r = iaa.Affine(rotate=(-20))     # rotate image 20 degrees in reverse
crop = iaa.Crop(px=(5, 32))              # Crop between 5 to 32 pixels
hflip = iaa.Fliplr(1)                    # horizontal flips for 100% of images
vflip = iaa.Flipud(1)                    # vertical flips for 100% of images
gblur = iaa.GaussianBlur(sigma=(1, 1.5)) # gaussian blur images with a sigma of 1.0 to 1.5
motionblur = iaa.MotionBlur(8)           # motion blur images with a kernel size 8

seq_rp = iaa.Sequential([
    iaa.Rot90((1,3)),                           # randomly rotate image from 90,180,270 degrees
    iaa.PerspectiveTransform(scale=(0.02, 0.1)) # Skews and transform images without black bg
])

seq_cfg = iaa.Sequential([
    iaa.Crop(px=(5, 32)),                      # crop images from each side by 5 to 32px (randomly chosen)
    iaa.Fliplr(0.5),                           # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 1.5))           # blur images with a sigma of 0 to 1.5
])

seq_fm = iaa.Sequential([
    iaa.Flipud(1),                            # vertical flips all the images
    iaa.MotionBlur(k=6)                       # motion blur images with a kernel size 6
])

if False:
    pipeline = []
    pipeline.append(rotate90)
    pipeline.append(rotate180)
    pipeline.append(rotate270)
    pipeline.append(perc_transform)
    pipeline.append(crop)
    pipeline.append(hflip)
    pipeline.append(vflip)
    pipeline.append(seq_rp)
    pipeline.append(seq_cfg)
    pipeline.append(seq_fm)
if True:
    pipeline = []
    pipeline.append(rotate90)
    pipeline.append(rotate180)
    pipeline.append(rotate270)
    pipeline.append(random_rotate)
    pipeline.append(perc_transform)
    pipeline.append(rotate10)
    pipeline.append(rotate10r)
    pipeline.append(rotate5)
    pipeline.append(rotate5r)
    pipeline.append(rotate3)
    pipeline.append(rotate3r)
    pipeline.append(rotate7)
    pipeline.append(rotate7r)
    pipeline.append(rotate15)
    pipeline.append(rotate15r)
    pipeline.append(rotate20)
    pipeline.append(rotate20r)
    pipeline.append(crop)
    pipeline.append(hflip)
    pipeline.append(vflip)
    pipeline.append(gblur)
    pipeline.append(motionblur)
    pipeline.append(seq_rp)
    pipeline.append(seq_cfg)
    pipeline.append(seq_fm)



import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, BatchNormalization, LeakyReLU, Add, Multiply, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, Permute


# Attention Block (optional, can be removed if not needed)
# SE Block for Channel Attention
def se_block(input_tensor, ratio=8):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]

    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([input_tensor, se])
    return x

# Spatial Attention Block
def spatial_attention_block(input_feature):
    kernel_size = 3

    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    spatial_attention_feature = Conv2D(1, kernel_size, strides=1, padding='same', activation='sigmoid', use_bias=False)(concat)
    spatial_attention_feature = Multiply()([input_feature, spatial_attention_feature])
    return spatial_attention_feature

def attention_block(x, g, inter_channels):
    """
    Simple Attention Block for shallow U-Net.
    """
    theta_x = Conv2D(inter_channels, kernel_size=2, strides=2, padding='same')(x)
    phi_g = Conv2D(inter_channels, kernel_size=1, padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    relu_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, kernel_size=1, padding='same')(relu_xg)
    sigmoid_psi = Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=(2, 2))(sigmoid_psi)
    attention_output = Multiply()([x, upsample_psi])
    return attention_output

# Updated U-Net with SE Attention
def unet_with_se_attention(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # Bottleneck with SE Attention
    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = BatchNormalization()(conv5)

    # Apply SE Attention
    se_conv5 = spatial_attention_block(conv5)

    # Decoder
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(se_conv5)
    merge6 = Concatenate()([up6, conv4])
    conv6 = Conv2D(256, (3, 3), padding='same')(merge6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(128, (3, 3), padding='same')(merge7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = Concatenate()([up8, conv2])
    conv8 = Conv2D(64, (3, 3), padding='same')(merge8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = Concatenate()([up9, conv1])
    conv9 = Conv2D(32, (3, 3), padding='same')(merge9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = BatchNormalization()(conv9)

    # Output Layer
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs, conv10)

    return model

# Input shape
input_shape = (target_width, target_height, 1)  # Example for grayscale images, modify if using RGB

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
numepochs = 80

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile`.
    model = unet_with_se_attention(input_shape)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

checkpoint_dir = './checkpoints'
checkpoint_prefix = checkpoint_dir + '/ckpt_{epochs}'

from tensorflow.keras.callbacks import EarlyStopping

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Create a checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,  # Save only the model weights
    save_best_only=False,  # Save every checkpoint, not just the best
    save_freq='epoch',  # Save at the end of every epoch
    verbose=1  # Verbosity mode, 1 = progress bar
)
# Set a memory limit for each GPU
# Summary of the model
model.summary()

model.load_weights(f'./checkpoints/ckpt_epochs_{numepochs}').expect_partial()

full_target_test = renormal_load_image_from_dir('../Data/Line/Clean')
full_noisy_test = renormal_load_image_from_dir('../Data/Line/Noisy')
psnrsk=[]
ssim=[]
rmse=[]
fsim=[]
uiq=[]
i=0
for testimage,  testimage2 in zip(full_noisy_test, full_target_test):
    testimage = np.expand_dims(testimage, axis=0)
    testimage2 = np.expand_dims(testimage2, axis=0)
    generatedimg = model(testimage, training=False, mask=None)

    i = i+1
    rows=1
    cols=3
    fig, ax = plt.subplots(1,3,figsize=(16,5), facecolor='white')
    fig.set_facecolor("white")
    fig.tight_layout()
    ax[0].imshow(testimage[0], cmap=plt.cm.gray, aspect='0.91')
    ax[2].imshow(testimage2[0], cmap=plt.cm.gray, aspect='0.91')
    ax[1].imshow(generatedimg[0], cmap=plt.cm.gray, aspect='0.91')
    #ax[0].imshow(testimage[0], cmap=plt.cm.gray)
    #ax[1].imshow(generatedimg[0], cmap=plt.cm.gray)
    #ax[2].imshow(testimage2[0], cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title("Noisy test image", fontsize = 18)
    ax[1].axis('off')
    ax[1].set_title("Clear test image", fontsize = 18)
    ax[2].axis('off')
    ax[2].set_title("Denoised test image", fontsize = 18)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig('./neweval_uattention_test_standard_%s.png'%(i))
    plt.close()

    plt.figure(figsize=(2.56,2.56), dpi=100)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0, hspace=0)
    plt.tight_layout()
    plt.imshow(testimage2[0], cmap=plt.cm.gray, aspect='0.91')
    plt.axis('off')
    plt.margins(0,0)
    plt.savefig('./targetx_uattention_test_%s.png'%(i), dpi=100, bbox_inches='tight', pad_inches = 0)
    plt.close()

    plt.figure(figsize=(2.56,2.56), dpi=100)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0, hspace=0)
    plt.tight_layout()
        #plt.imshow(predout, cmap=plt.cm.gray, aspect='0.78125')
    plt.imshow(generatedimg[0], cmap=plt.cm.gray, aspect='0.91')
    plt.axis('off')
    plt.margins(0,0)
    plt.savefig('./predout_uattention_test_%s.png'%(i), dpi=100, bbox_inches='tight', pad_inches = 0)
    plt.close()
##Metrics
    testimage2  = (testimage2.reshape(1, target_height,target_width,1))
    X_test2_fig  = testimage2.reshape(-1,target_height,target_width,1)

    X_test1_fig  = testimage.reshape(-1,target_height,target_width,1)

    preds1=generatedimg


    targetx = X_test2_fig.reshape(target_height,target_width,1)
    testx = X_test1_fig.reshape(target_height,target_width,1)
    predout = preds1.reshape(target_height,target_width,1)


    targetimg = testimage2[0]
    targetx = targetimg
    print("Target=", targetx.shape)

    predimg = generatedimg[0].numpy()
    predout = predimg
    print("Predict=", predimg.shape)
    # Convert images to float format
    targetimg = img_as_float(targetimg)
    predimg = img_as_float(predimg)

    psnr_denoised = peak_signal_noise_ratio(targetimg, predimg, data_range=targetimg.max() - targetimg.min())
    ssim_denoised = compssim(np.squeeze(targetx), np.squeeze(predout), data_range=targetimg.max() - targetimg.min(), multichannel=False)
    rmse_denoised = evaluation(targetx, predout, metrics=["rmse"])
    fsim_denoised = evaluation(targetx, predout, metrics=["fsim"])
    uiq_denoised = evaluation(targetx, predout, metrics=["uiq"])
    print('PSNR=',psnr_denoised)
    psnrsk.append(psnr_denoised)
    ssim.append(ssim_denoised)
    rmse.append(rmse_denoised)
    fsim.append(fsim_denoised)
    uiq.append(uiq_denoised)
avgpsnr = np.mean(psnrsk)
stdpsnr = np.std(psnrsk)
avgssim = np.mean(ssim)
stdssim = np.std(ssim)
avgfsim = np.mean(fsim)
stdfsim = np.std(fsim)
avguiq = np.mean(uiq)
stduiq = np.std(uiq)
avgrmse = np.mean(rmse)
stdrmse = np.std(rmse)
print('psnr=', avgpsnr, 'ssim=', avgssim, 'fsim=', avgfsim, 'uiq=', avguiq, 'rmse=', avgrmse)
print('psnrstd=', stdpsnr, 'ssimstd=', stdssim, 'fsimstd=', stdfsim, 'uiqstd=', stduiq, 'rmsestd=', stdrmse)
exit()
