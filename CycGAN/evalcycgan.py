## For the evaluation of DnCNN
##
## Seoleun Shin (KRISS), 2023-June
#
## Use metrics based on https://github.com/up42/image-similarity-measures
#  and modified for our samples (Seoleun Shin, 2023/June/01) ##
##---------------------------------------------------------------------------------------------------------
##

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import re
import os
import math
import random
import cv2
from PIL import Image
import imageio
import shutil
import glob
from IPython import get_ipython
#from keras.preprocessing import image
import skimage
from skimage import color
from tensorflow.keras.utils import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compssim
from image_similarity_measures.evaluate import evaluation
import pickle


os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'


tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available()



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


target_height = 512
target_width = 512
np.random.seed(456)
image_size = 512



BATCH_SIZE =  1
def parse_function(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = (tf.cast(image,tf.float32)/ 127.5) - 1
    image = tf.reshape(image, [256, 256, 3])
    return image
def data_augment(image):
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    if p_crop > 0.5:
        image = tf.image.resize(image, [286, 286])
        image = tf.image.random_crop(image, size=[256, 256, 3])
        if p_crop > 0.9:
            image = tf.image.resize(image, [300, 300])
            image = tf.image.random_crop(image, size=[256, 256, 3])  
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
num_parallel_calls=tf.data.experimental.AUTOTUNE
def getSet(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls)
    dataset = dataset.map(data_augment, num_parallel_calls)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(num_parallel_calls)
    return dataset


OUTPUT_CHANNELS = 3
def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    result = keras.Sequential()
    result.add(layers.Conv2D(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
    result.add(layers.LeakyReLU())
    return result



def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result



def Generator():
    inputs = layers.Input(shape=[512,512,3])
    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64) 256/2（padding same，(256-4+1)/2）
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024) 1*2（padding same，(1*2-4+1)），，
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS,4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh') # (bs, 256, 256, 3)，，
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    x = last(x)
    return keras.Model(inputs=inputs, outputs=x)



Generator().summary()


tf.keras.utils.plot_model(Generator(), show_shapes=True, dpi=64)



def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    inp = layers.Input(shape=[512, 512, 3], name='input_image')
    x = inp
    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)
    leaky_relu = layers.LeakyReLU()(norm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    last = layers.Conv2D(1,4,strides=1,kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)，，
    return tf.keras.Model(inputs=inp, outputs=last)


Discriminator().summary()


tf.keras.utils.plot_model(Discriminator(),show_shapes=True, dpi=64)


class CycleGan(keras.Model):
    def __init__(self,monet_generator,photo_generator,monet_discriminator,photo_discriminator,lambda_cycle=15):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
    def compile(self,m_gen_optimizer,p_gen_optimizer,m_disc_optimizer,p_disc_optimizer,gen_loss_fn,disc_loss_fn,cycle_loss_fn,identity_loss_fn):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer 
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn  
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        with tf.GradientTape(persistent=True) as tape:
            fake_monet = self.m_gen(real_photo, training=True)       #G(x)
            cycled_photo = self.p_gen(fake_monet, training=True)     #F(G(x))
            fake_photo = self.p_gen(real_monet, training=True)       #F(y)
            cycled_monet = self.m_gen(fake_photo, training=True)     #G(F(y))
            same_monet = self.m_gen(real_monet, training=True)       #G(y)
            same_photo = self.p_gen(real_photo, training=True)       #F(x)
            disc_real_monet = self.m_disc(real_monet, training=True) #DY(y)
            disc_real_photo = self.p_disc(real_photo, training=True) #DX(x)
            disc_fake_monet = self.m_disc(fake_monet, training=True) #DY(G(x))
            disc_fake_photo = self.p_disc(fake_photo, training=True) #DX(F(y))
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)
            total_cycle_loss=self.cycle_loss_fn(real_monet,cycled_monet,self.lambda_cycle)+self.cycle_loss_fn(real_photo,cycled_photo,self.lambda_cycle)
            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss+total_cycle_loss+self.identity_loss_fn(real_monet, same_monet,self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss+total_cycle_loss+self.identity_loss_fn(real_photo, same_photo,self.lambda_cycle)
            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,self.p_gen.trainable_variables)
        monet_discriminator_gradients = tape.gradient(monet_disc_loss,self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,self.p_disc.trainable_variables)
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,self.m_gen.trainable_variables))
        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,self.p_gen.trainable_variables))
        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,self.m_disc.trainable_variables))
        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,self.p_disc.trainable_variables))
        
        '''if (iteration + 1) % sample_interval == 0:

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(generator)'''
        #https://github.com/GANs-in-Action/gans-in-action/blob/master/chapter-3/Chapter_3_GAN.ipynb
        
        return {"monet_gen_loss": total_monet_gen_loss,"photo_gen_loss": total_photo_gen_loss,"monet_disc_loss": monet_disc_loss,"photo_disc_loss": photo_disc_loss}



def discriminator_loss(real, generated):
    real_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real),real)
    generated_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated),generated)
    total_disc_loss=real_loss+generated_loss
    return total_disc_loss*0.5



def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated),generated)



def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1



def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss



monet_generator_optimizer = tf.keras.optimizers.Adamax(2e-4, beta_1=0.5)
photo_generator_optimizer = tf.keras.optimizers.Adamax(2e-4, beta_1=0.5)
monet_discriminator_optimizer = tf.keras.optimizers.Adamax(2e-4, beta_1=0.5)
photo_discriminator_optimizer = tf.keras.optimizers.Adamax(2e-4, beta_1=0.5)



monet_generator=Generator()
photo_generator=Generator()
monet_discriminator=Discriminator()
photo_discriminator=Discriminator()
cycle_gan_model = CycleGan(monet_generator,photo_generator,monet_discriminator,photo_discriminator)
cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss)


### LOAD the Model for Evaluation

import os.path
fname='./Gen_IMG/CycleGAN/wgen_monet1.h5'
if os.path.isfile(fname):
    monet_generator.load_weights(fname)
fname='./Gen_IMG/CycleGAN/wgen_photo1.h5'
if os.path.isfile(fname):
    photo_generator.load_weights(fname)
fname='./Gen_IMG/CycleGAN/wdisc_monet1.h5'
if os.path.isfile(fname):
    monet_discriminator.load_weights(fname)
fname='./Gen_IMG/CycleGAN/wdisc_photo1.h5'
if os.path.isfile(fname):
    photo_discriminator.load_weights(fname)


#from tensorflow.keras.callbacks import EarlyStopping
#early_stop=EarlyStopping(monitor='generator_loss',mode='min',patience=1,restore_best_weights=True)



gen_monet = tf.keras.models.load_model('./Gen_IMG/CycleGAN/gen_monet1.h5')
gen_photo = tf.keras.models.load_model('./Gen_IMG/CycleGAN/gen_photo1.h5')
disc_monet = tf.keras.models.load_model('./Gen_IMG/CycleGAN/disc_monet1.h5')
disc_photo = tf.keras.models.load_model('./Gen_IMG/CycleGAN/disc_photo1.h5')



def renormal_load_image_from_dir2(img_path):
    #file_list = glob.glob(img_path)
    file_list = glob.glob(img_path+'/*.tif')
    file_list.sort()
    img_list = np.empty((len(file_list), image_size, image_size, 3))
    for i, fig in enumerate(file_list):
        figures = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
        figures = cropImage(figures,(0,0),(512,440))
        height = tf.shape(figures)[0]
        width = tf.shape(figures)[1]
        crop_size = tf.minimum(height, width)
        img_array = tf.keras.utils.img_to_array(figures).astype('uint8')
        #img_array = tf.keras.utils.img_to_array(figures).astype('float32')
        img_array = tf.image.resize(img_array, size=[image_size, image_size], antialias=True)/127.5 -1
        img_list[i] = img_array
    return img_list


cycleganmodel = gen_photo


psnrsk=[]
ssim=[]
rmse=[]
fsim=[]
uiq=[]
i = 0
file_list1  = renormal_load_image_from_dir2('../Data/Line/Noisy')
file_list2  = renormal_load_image_from_dir2('../Data/Line/Clean')
for file1, file2 in zip(file_list1, file_list2):
    X_test1_fig  = (file1.reshape(-1, target_height,target_width,3)*127.5 + 127.5).astype(np.uint8)
    file1  = (file1.reshape(1, target_height,target_width,3))
    X_test_target1 = (file2.reshape(-1, target_height,target_width,3)*127.5+127.5).astype(np.uint8)

    preds1=cycleganmodel.predict(file1,batch_size=1)
    preds1  = (preds1.reshape(-1, target_height,target_width,3)*127.5+127.5).astype(np.uint8)

    targetx = X_test_target1.reshape(target_height,target_width,3)
    testx = X_test1_fig.reshape(target_height,target_width,3)
    predout = preds1.reshape(target_height,target_width,3)
    print('CHECK1=',np.shape(targetx))
    print('CHECK2=',np.shape(testx))
    print('CHECK3=',np.shape(predout))

    psnrsk_noise, psnrsk_denoised = peak_signal_noise_ratio(targetx, testx), peak_signal_noise_ratio(targetx, predout)
    psnrsk.append(psnrsk_denoised)

    ssim_noise, ssim_denoised = compssim(targetx, testx, multichannel = True), compssim(targetx, predout, multichannel = True)
    ssim.append(ssim_denoised)

    rmse_noise, rmse_denoised = evaluation(targetx, testx, metrics=["rmse"]), evaluation(targetx, predout, metrics=["rmse"])
    rmse.append(rmse_denoised)

    fsim_noise, fsim_denoised = evaluation(targetx, testx, metrics=["fsim"]), evaluation(targetx, predout, metrics=["fsim"])
    fsim.append(fsim_denoised)
    uiq_noise, uiq_denoised = evaluation(targetx, testx, metrics=["uiq"]), evaluation(targetx, predout, metrics=["uiq"])
    uiq.append(uiq_denoised)

    i=i+1
    rows=1
    cols=3
    fig, ax = plt.subplots(1,3,figsize=(16,5), facecolor='white')
    fig.tight_layout()
    fig.set_facecolor("white")
    ax[0].imshow(testx[:,:,0], cmap=plt.cm.gray, aspect='0.859375')
    ax[0].axis('off')
    ax[0].set_title("Noisy test image", fontsize = 18)
    ax[1].imshow(targetx[:,:,0], cmap=plt.cm.gray, aspect='0.859375')
    ax[1].axis('off')
    ax[1].set_title("Clear test image", fontsize = 18)
    #ax[2].imshow(preds1.reshape(target_height,target_width,3), cmap=plt.cm.gray, aspect='0.859375')
    ax[2].imshow(predout[:,:,0], cmap=plt.cm.gray, aspect='0.859375')
    ax[2].axis('off')
    ax[2].set_title("Denoised test image", fontsize = 18)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    #fig.savefig('./neweval_cycgan_test_standard_%s.png'%(i))

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
