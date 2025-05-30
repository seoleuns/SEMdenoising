## 
# Denoising SEM images with cycleGAN
# 
# In-Ho Lee, KRISS, December 20, 2021.
# Seoleun Shin, KRISS, upgraded 2023 Jan.
#
# https://www.kaggle.com/upamanyumukherjee/cyclegan
# https://machinelearningmastery.com/what-is-cyclegan/
#
#
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

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()



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

    img = cv2.imread(filename)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = rgba[:,:,3]
    rgb_channels = image_4channel[:,:,:3]


    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = white + base
    return final_image.astype(np.uint8)



target_width = 512
target_height = 416
np.random.seed(456)


img_path='../Data/time/high'
file_list = glob.glob(img_path+'/30us*.tif')
img_path='../Data/time/low'
gile_list = glob.glob(img_path+'/100ns*.tif')

img_path='../Data/Line/Clean'
kile_list = glob.glob(img_path+'/*.tif')
img_path='../Data/Line/Noisy'
mile_list = glob.glob(img_path+'/*.tif')

# You need to clean-up jpg files in this directory whenever run this code or after this step remove the earlier part 
# you can try this or just remove files manually
#img_path='./Gen_IMG/CycleGAN/train'
#file_list = glob.glob(img_path+'/*.jpg')
#for i in file_list:
#    os.remove(i)
#img_path='./Gen_IMG/CycleGAN/train_cleaned'
#file_list = glob.glob(img_path+'/*.jpg')
#for i in file_list:
#    os.remove(i)
#img_path='./Gen_IMG/CycleGAN/test'
#file_list = glob.glob(img_path+'/*.jpg')
#for i in file_list:
#    os.remove(i)
#img_path='./Gen_IMG/CycleGAN/test_cleaned'
#file_list = glob.glob(img_path+'/*.jpg')
#for i in file_list:
#    os.remove(i)


k1=0
for i in file_list:
    ii = "./Gen_IMG/CycleGAN/train_cleaned/"+str(k1)+'.tif'
    shutil.copyfile(i,ii)
    k1=k1+1 

k1=0
for i in gile_list:
    ii = "./Gen_IMG/CycleGAN/train/"+str(k1)+'.tif'
    shutil.copyfile(i,ii)
    k1=k1+1 

k1=0
for i in kile_list:
    ii = "./Gen_IMG/CycleGAN/test_cleaned/"+str(k1)+'.tif'
    shutil.copyfile(i,ii)
    k1=k1+1 

k1=0
for i in mile_list:
    ii = "./Gen_IMG/CycleGAN/test/"+str(k1)+'.tif'
    shutil.copyfile(i,ii)
    k1=k1+1 

img_path='./Gen_IMG/CycleGAN/train/'
file_list = glob.glob(img_path+'*.tif')
for i in file_list:
    j=i ; j=j.replace('.tif','.jpg')
    print(i,j)
    img0=read_transparent_png(i)
    img=cropImage(img0,(3,3),(256,256))
    imageio.imwrite(j,img)
    if True:
        img1=cropImage(img0,(120,120),(256,256))
        j1=i ; k=random.randint(1000,100000) ; k1=random.randint(1000,100000)
        k=str(k)+str(k1)+'.jpg' ; j1=j1.replace('.tif',k)
        imageio.imwrite(j1,img1)

img_path='./Gen_IMG/CycleGAN/train_cleaned/'
file_list = glob.glob(img_path+'*.tif')
for i in file_list:
    j=i ; j=j.replace('.tif','.jpg')
    img0=read_transparent_png(i)
    img=cropImage(img0,(3,3),(256,256))
    imageio.imwrite(j,img)
    if True:
        img1=cropImage(img0,(120,120),(256,256))
        j1=i ; k=random.randint(1000,100000) ; k1=random.randint(1000,100000)
        k=str(k)+str(k1)+'.jpg' ; j1=j1.replace('.tif',k)
        imageio.imwrite(j1,img1)

img_path='./Gen_IMG/CycleGAN/test/'
file_list = glob.glob(img_path+'*.tif')
for i in file_list:
    j=i ; j=j.replace('.tif','.jpg')
    img0=read_transparent_png(i)
    img=cropImage(img0,(3,3),(256,256))
    imageio.imwrite(j,img)
    if True:
        img1=cropImage(img0,(120,120),(256,256))
        j1=i ; k=random.randint(1000,100000) ; k1=random.randint(1000,100000)
        k=str(k)+str(k1)+'.jpg' ; j1=j1.replace('.tif',k)
        imageio.imwrite(j1,img1)
img_path='./Gen_IMG/CycleGAN/test_cleaned/'
file_list = glob.glob(img_path+'*.tif')
for i in file_list:
    j=i ; j=j.replace('.tif','.jpg')
    img0=read_transparent_png(i)
    img=cropImage(img0,(3,3),(256,256))
    imageio.imwrite(j,img)
    if True:
        img1=cropImage(img0,(120,120),(256,256))
        j1=i ;  k=random.randint(1000,100000) ; k1=random.randint(1000,100000)
        k=str(k)+str(k1)+'.jpg' ; j1=j1.replace('.tif',k)
        imageio.imwrite(j1,img1)


img_path='./Gen_IMG/CycleGAN/train'
file_list = glob.glob(img_path+'/*.jpg')
for i in file_list:
    j1=i ; k=random.randint(1000,100000) ; k1=random.randint(1000,100000)
    k=str(k)+str(k1)+'.jpg' ; j1=j1.replace('.jpg',k)
    img = cv2.imread(i)
    empty_img=rotateImage(img,90)
    cv2.imwrite(j1, empty_img)
    j1=i ; k=random.randint(1000,100000) ; k1=random.randint(1000,100000)
    k=str(k)+str(k1)+'.jpg' ; j1=j1.replace('.jpg',k)
    empty_img =rotateImage(img,180)
    cv2.imwrite(j1, empty_img)
    j1=i ; k=random.randint(1000,100000) ; k1=random.randint(1000,100000)
    k=str(k)+str(k1)+'.jpg' ;  j1=j1.replace('.jpg',k)
    empty_img=rotateImage(img,270)
    cv2.imwrite(j1, empty_img)


img_path='./Gen_IMG/CycleGAN/train'
file_list = glob.glob(img_path+'/*.tif')
for i in file_list:
    os.remove(i)
img_path='./Gen_IMG/CycleGAN/train_cleaned'
file_list = glob.glob(img_path+'/*.tif')
for i in file_list:
    os.remove(i)
img_path='./Gen_IMG/CycleGAN/test'
file_list = glob.glob(img_path+'/*.tif')
for i in file_list:
    os.remove(i)            
img_path='./Gen_IMG/CycleGAN/test_cleaned'
file_list = glob.glob(img_path+'/*.tif')
for i in file_list:
    os.remove(i) 



GCS_PATH = './Gen_IMG/CycleGAN'
fn_monet = tf.io.gfile.glob(str(GCS_PATH + '/train/*.jpg'))
fn_photo = tf.io.gfile.glob(str(GCS_PATH + '/train_cleaned/*.jpg'))
fn_monet1 = tf.io.gfile.glob(str(GCS_PATH + '/test/*.jpg'))
fn_photo1 = tf.io.gfile.glob(str(GCS_PATH + '/test_cleaned/*.jpg'))


# View one image
import imageio
photo_image_names =os.listdir('./Gen_IMG/CycleGAN/train_cleaned')
photo_img =imageio.imread(os.path.join('./Gen_IMG/CycleGAN/train_cleaned',photo_image_names[10]))
plt.imshow(photo_img)
plt.figure()


# View one image
import imageio
monet_image_names = os.listdir('./Gen_IMG/CycleGAN/train')
monet_img = imageio.imread(os.path.join('./Gen_IMG/CycleGAN/train',monet_image_names[10]))
plt.imshow(monet_img)
plt.figure()


if True:
    rand_monet = r"./Gen_IMG/CycleGAN/train/2.jpg"
    rand_photo = r"./Gen_IMG/CycleGAN/train_cleaned/2.jpg"



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
monet_ds=getSet(fn_monet)
photo_ds=getSet(fn_photo)


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
    inputs = layers.Input(shape=[256,256,3])
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
    inp = layers.Input(shape=[256, 256, 3], name='input_image')
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
        #real_monet y，real_photo x，m_gen G，p_gen F，p_disc DX，m_disc DY
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



with strategy.scope():
    def discriminator_loss(real, generated):
        real_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real),real)
        generated_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated),generated)
        total_disc_loss=real_loss+generated_loss
        return total_disc_loss*0.5



with strategy.scope():
    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated),generated)



with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return LAMBDA * loss1



with strategy.scope():
    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss



with strategy.scope():
    monet_generator_optimizer = tf.keras.optimizers.Adamax(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adamax(2e-4, beta_1=0.5)
    monet_discriminator_optimizer = tf.keras.optimizers.Adamax(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adamax(2e-4, beta_1=0.5)

global im_to_gif
im_to_gif = np.zeros((300,256,256,3))
photo = next(iter(monet_ds))
num_photo = 0
plt.imshow(photo[num_photo]*0.5 + 0.5)


class GANMonitor(keras.callbacks.Callback):
   """A callback to generate and save images after each epoch"""
   def on_epoch_end(self,epoch,logs=None):
        prediction =  monet_generator(photo,training=False)[num_photo].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)



with strategy.scope():
    monet_generator=Generator()
    photo_generator=Generator()
    monet_discriminator=Discriminator()
    photo_discriminator=Discriminator()
with strategy.scope():
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
plotter = GANMonitor()



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



from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='generator_loss',mode='min',patience=1,restore_best_weights=True)



#tf.keras.utils.plot_model(cycle_gan_model,show_shapes=True, dpi=64)



get_ipython().run_cell_magic('time', '', 'steps_per_epoch=(max(len(fn_monet),len(fn_photo)))//BATCH_SIZE\nhistory=cycle_gan_model.fit(tf.data.Dataset.zip((monet_ds, photo_ds)),epochs=200,steps_per_epoch=steps_per_epoch,callbacks=[History(),plotter])')


monet_generator.save_weights('./Gen_IMG/CycleGAN/wgen_monet1.h5')
photo_generator.save_weights('./Gen_IMG/CycleGAN/wgen_photo1.h5')
monet_discriminator.save_weights('./Gen_IMG/CycleGAN/wdisc_monet1.h5')
photo_discriminator.save_weights('./Gen_IMG/CycleGAN/wdisc_photo1.h5')


monet_generator.save('./Gen_IMG/CycleGAN/gen_monet1.h5')
photo_generator.save('./Gen_IMG/CycleGAN/gen_photo1.h5')
monet_discriminator.save('./Gen_IMG/CycleGAN/disc_monet1.h5')
photo_discriminator.save('./Gen_IMG/CycleGAN/disc_photo1.h5')
import pickle


with open('./Gen_IMG/CycleGAN/history1.pkl','wb') as f:
    pickle.dump(history.history, f)


gen_monet = tf.keras.models.load_model('./Gen_IMG/CycleGAN/gen_monet1.h5')
gen_photo = tf.keras.models.load_model('./Gen_IMG/CycleGAN/gen_photo1.h5')
disc_monet = tf.keras.models.load_model('./Gen_IMG/CycleGAN/disc_monet1.h5')
disc_photo = tf.keras.models.load_model('./Gen_IMG/CycleGAN/disc_photo1.h5')


def plot_acc_and_loss(history, load=False):
    monet_g = []
    photo_g = []
    monet_d = []
    photo_d = []
    if load==True:
        for i in range(np.array(history["monet_gen_loss"]).shape[0]):
            monet_g.append(np.array(history["monet_gen_loss"][i]).squeeze().mean())
            photo_g.append(np.array(history["photo_gen_loss"][i]).squeeze().mean())
            monet_d.append(np.array(history["monet_disc_loss"][i]).squeeze().mean())
            photo_d.append(np.array(history["photo_disc_loss"][i]).squeeze().mean())
    else:
        for i in range(np.array(history.history["monet_gen_loss"]).shape[0]):
            monet_g.append(np.array(history.history["monet_gen_loss"][i]).squeeze().mean())
            photo_g.append(np.array(history.history["photo_gen_loss"][i]).squeeze().mean())
            monet_d.append(np.array(history.history["monet_disc_loss"][i]).squeeze().mean())
            photo_d.append(np.array(history.history["photo_disc_loss"][i]).squeeze().mean())
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(monet_g,label="Monet")
    axs[0].plot(photo_g,label="Photo")
    axs[0].set_title("generator loss")
    axs[0].legend()
    axs[1].plot(monet_d, label="Monet")
    axs[1].plot(photo_d,label="Photo")
    axs[1].set_title("discriminator loss")
    axs[1].legend()
    plt.show()



import PIL
def gen_input_img(num_photo=0,load=False):
    fig, ax = plt.subplots(figsize=(5,5))
    if load == True:
        img = np.array(PIL.Image.open('./Gen_IMG/CycleGAN/train/2.jpg'))
        plt.imshow(img)
        ax.axis("off")
    else:
        img = photo[3]*0.5 + 0.5
        plt.imshow(img)
        ax.axis("off")
        plt.title('Input photo')    


gen_input_img(num_photo=0,load=True)


import PIL
from IPython import display
import imageio
import shutil


gen_input_img(num_photo=num_photo,load=True)



if False:
    import tensorflow_docs.vis.embed as embed



# Prediction evolution according to epoch
if False:
    embed.embed_file(anim_file)


testset = tf.data.Dataset.from_tensor_slices(fn_monet1)
testset = testset.shuffle(len(fn_photo))
testset = testset.map(parse_function, num_parallel_calls)
testset=testset.batch(1)
testset = testset.prefetch(num_parallel_calls)
_, ax = plt.subplots(5, 2, figsize=(20, 20))
for i, img in enumerate(testset.take(5)):
    prediction = monet_generator(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input Photo")
    ax[i, 1].set_title("Noisy Photo")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.savefig('Test_input_to_noisy.png')
#plt.show()
plt.close()



def evaluate_cycle(ds, generator_a, generator_b, n_samples=1):
    fig, axes = plt.subplots(n_samples, 3, figsize=(22, (n_samples*6)))
    axes = axes.flatten()
    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        idx = n_sample*3
        example_sample = next(ds_iter)
        generated_a_sample = generator_a.predict(example_sample)
        generated_b_sample = generator_b.predict(generated_a_sample)
        #axes[idx].set_title('Input image', fontsize=18)
        axes[idx].set_title('Input Noisy image', fontsize=18)
        axes[idx].imshow(example_sample[0] * 0.5 + 0.5)
        axes[idx].axis('off')
        #axes[idx+1].set_title('Generated Noisy image', fontsize=18)
        axes[idx+1].set_title('Generated Clear image', fontsize=18)
        axes[idx+1].imshow(generated_a_sample[0] * 0.5 + 0.5)
        axes[idx+1].axis('off')
        #axes[idx+2].set_title('Cycled Denoised image', fontsize=18)
        axes[idx+2].set_title('Generated Cycled image', fontsize=18)
        axes[idx+2].imshow(generated_b_sample[0] * 0.5 + 0.5)
        axes[idx+2].axis('off')
    plt.savefig('Eval_cycle_noisy_to_clear.png')
    plt.show()
    plt.close()

evaluate_cycle(testset.take(3), photo_generator, monet_generator, n_samples=3)


import PIL
def predict_and_save(input_ds, generator_model, output_path):
    i = 1
    for img in input_ds:
        prediction = generator_model(img, training=False)[0].numpy() # make predition
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)   # re-scale
        im = PIL.Image.fromarray(prediction)
        if i > 9000:
           im.save(output_path + str(i) + ".jpg")
        i += 1
