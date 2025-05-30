# Denoising SEM images with autoencoder (Image augmentation)
# 
# In-Ho Lee, KRISS, August 22, 2021.
# https://keras.io/examples/vision/oxford_pets_image_segmentation/
# 
# Seoleun Shin, KRISS, 2023-Jan-01
# Modified referring to https://keras.io/examples/vision/autoencoder
##--------------------------------------------------------------------

#If Use only CPU
if False:
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


import tensorflow as tf 
tf.test.is_gpu_available()


tf.config.list_physical_devices('GPU')



import glob 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from keras.preprocessing import image
from keras.models import Model, load_model
#from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, LeakyReLU
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
from keras import regularizers
import keras.backend as kb
from tensorflow.keras import layers
import shutil
import os
from IPython import display
from IPython import get_ipython


import tensorflow as tf 
tf.test.is_gpu_available()


from PIL import Image
import imageio


target_width = 512
target_height = 512
image_size =512

np.random.seed(456)



get_ipython().run_line_magic('matplotlib', 'inline')



plt.rcParams['figure.figsize'] = (10.0, 5.0)         # set default size of plots


trainpath='../Data/time/low'
traintargetpath='../Data/time/high'
testpath='../Data/Line/Noisy'
testtargetpath='../Data/Line/Clean'


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


# In[19]:


full_train  = renormal_load_image_from_dir(trainpath)
full_target = renormal_load_image_from_dir(traintargetpath)


# In[20]:


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


# In[21]:


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


# In[22]:


get_ipython().run_cell_magic('time', '', 'processed_train  = augment_pipeline(pipeline, full_train.reshape(-1,target_height,target_width))\nprocessed_target = augment_pipeline(pipeline, full_target.reshape(-1,target_height,target_width))\nprocessed_train  = processed_train.reshape(-1,target_height,target_width,1)\nprocessed_target = processed_target.reshape(-1,target_height,target_width,1)\nprocessed_train.shape')



input_layer = Input(shape=(None,None,1))
# encoder
e = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
e = BatchNormalization()(e)
e = Conv2D(128, (3, 3), activation='relu', padding='same')(e)
e = BatchNormalization()(e)
e = Conv2D(256, (3, 3), activation='relu', padding='same')(e)
for _ in range(3):
    e = Conv2D(256, (3, 3), activation='relu', padding='same')(e)
e = MaxPooling2D((2, 2), padding='same')(e)
# decoder
d = Conv2D(256, (3, 3), activation='relu', padding='same')(e)
for _ in range(3):
    d = Conv2D(256, (3, 3), activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2D(128, (3, 3), activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = UpSampling2D((2, 2))(d)
d = Conv2D(64, (3, 3), activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2D(128, (3, 3), activation='relu', padding='same')(d)
output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)



def my_loss(y_true, y_pred):
    alph=0.7
    squared_difference = tf.square(y_true - y_pred)
    if alph > 0.:
        return alph*tf.reduce_mean(squared_difference, axis=-1)+(1.-alph)*(1.-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0), axis=-1))
    else :
        beta=-alph
        return beta*tf.reduce_mean(squared_difference, axis=-1)+(1.-beta)*(1.-tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred,1.0), axis=-1))
# Loss function 
def ssim_loss(y_true, y_pred):
    return 1.-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
def msssim_loss(y_true, y_pred):
    return 1.-tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))


AEmodel = Model(input_layer,output_layer)
AEmodel.compile( optimizer='adam', loss='mse', metrics=['mse'])
AEmodel.summary()


early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=2, verbose=1, mode='auto')
checkpoint1 = ModelCheckpoint('best_val_loss.h5', monitor='val_loss', save_best_only=True)
checkpoint2 = ModelCheckpoint('best_loss.h5', monitor='loss', save_best_only=True)


#ForTest
get_ipython().run_cell_magic('time', '', 'history = AEmodel.fit(processed_train, processed_target, batch_size=1, epochs=40, verbose=1,\n                        validation_split=0.2,\n#                       validation_data=(val, target_val),\n                        callbacks=[checkpoint2])')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc='upper left')
plt.show()


plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc='upper left')
plt.show()


AEmodel.save('AutoEncoderModelFull.h5')

import keras
#ForTest
if False:
    AEmodel = keras.models.load_model('AutoEncoderModelFull.h5')


full_train_preds = AEmodel.predict(full_train, batch_size=1)
if False:
    AEmodel.load_weights('best_loss.h5')
    AEmodel.compile( optimizer='adam',loss=my_loss, metrics=['mse'])
train_preds = AEmodel.predict(full_train, batch_size=1)



AEmodel.evaluate(full_train, full_target, batch_size=1)


if False:
    AEmodel = keras.models.load_model('AutoEncoderModelBestLoss.h5')



full_train=full_train*255.
full_target=full_target*255.
full_train_preds=full_train_preds*255.



iidd=11
jjdd=15
fig, ax = plt.subplots(2,3,figsize=(22,16))
ax[0][0].imshow(full_train[iidd].reshape(target_height,target_width), cmap=plt.cm.gray)
ax[0][1].imshow(full_target[iidd].reshape(target_height,target_width), cmap=plt.cm.gray)
ax[0][2].imshow(full_train_preds[iidd].reshape(target_height,target_width), cmap=plt.cm.gray)
ax[1][0].imshow(full_train[jjdd].reshape(target_height,target_width), cmap=plt.cm.gray)
ax[1][1].imshow(full_target[jjdd].reshape(target_height,target_width), cmap=plt.cm.gray)
ax[1][2].imshow(full_train_preds[jjdd].reshape(target_height,target_width), cmap=plt.cm.gray)



X_test  = renormal_load_image_from_dir(testpath)
X_test  = X_test.reshape(-1,target_height,target_width,1)

X_test_target  = renormal_load_image_from_dir(testtargetpath)
X_test_target  = X_test_target.reshape(-1,target_height,target_width,1)

preds=AEmodel.predict(X_test,batch_size=1)

X_test=X_test*255.
preds=preds*255.



def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    plt.figure(figsize=(2*n*1.5, 2*len(args)*1.5))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i*n + j + 1)
            plt.imshow(args[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
plot_digits(X_test[:2], preds[:2])



def im_show(im_name):
    plt.figure(figsize=(20,8))
    img = cv2.imread(im_name, 0)
    plt.imshow(img, cmap="gray")
    plt.show(block=True)
    print(f"[INFO] Image shape: {img.shape} ")

