## For the evaluation of DnCNN
##
## Seoleun Shin (KRISS), 2023-June
#
## Use metrics based on https://github.com/up42/image-similarity-measures
#  and modified for our samples (Seoleun Shin, 2023/June/01) ##
##---------------------------------------------------------------------------------------------------------
##

#Use only CPU
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
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compssim
from image_similarity_measures.evaluate import evaluation
import tensorflow as tf 

tf.test.is_gpu_available()


from PIL import Image
import imageio
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



target_height = 512
target_width = 512
np.random.seed(456)
image_size = 512


def renormal_load_image_from_dir2(img_path):
    file_list = glob.glob(img_path+'/*.tif')
    file_list.sort()
    img_list = np.empty((len(file_list), target_height, target_width, 1))
    for i, fig in enumerate(file_list):
        figures = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
        figures = cropImage(figures,(0,0),(512,440))
        height = tf.shape(figures)[0]
        width = tf.shape(figures)[1]
        crop_size = tf.minimum(height, width)
        img_array = tf.keras.utils.img_to_array(figures).astype('float32')
        img_array = tf.image.resize(img_array, size=[target_height, target_width], antialias=True)
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
AEmodel.compile( optimizer='adam', metrics=['mse'])


early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=2, verbose=1, mode='auto')
checkpoint1 = ModelCheckpoint('best_val_loss.h5', monitor='val_loss', save_best_only=True)
checkpoint2 = ModelCheckpoint('best_loss.h5', monitor='loss', save_best_only=True)


import keras
#ForTest
if True:
    AEmodel = keras.models.load_model('AutoEncoderModelFull.h5')


psnrsk=[]
ssim=[]
rmse=[]
fsim=[]
uiq=[]
i = 0
file_list1  = renormal_load_image_from_dir2('../Data/Line/Noisy')
file_list2  = renormal_load_image_from_dir2('../Data/Line/Clean')

for file1, file2 in zip(file_list1, file_list2):
    file1  = (file1.reshape(1, target_height,target_width,1))
    X_test1_fig  = file1.reshape(-1,target_height,target_width,1)

    X_test_target1  = file2.reshape(-1,target_height,target_width,1)

    preds1=AEmodel.predict(file1,batch_size=1)

    X_test1_fig=X_test1_fig*255.
    X_test_target1_fig=X_test_target1*255.
    preds1_fig=preds1*255.

    targetx = X_test_target1.reshape(target_height,target_width,1)
    testx = X_test1_fig.reshape(target_height,target_width,1)
    predout = preds1.reshape(target_height,target_width,1)
    print('CHECK1=',np.shape(targetx))
    print('CHECK2=',np.shape(testx))
    print('CHECK3=',np.shape(predout))

    psnrsk_noise, psnrsk_denoised = peak_signal_noise_ratio(targetx, testx), peak_signal_noise_ratio(targetx, predout)
    psnrsk.append(psnrsk_denoised)
    ssim_noise, ssim_denoised = compssim(np.squeeze(targetx), np.squeeze(testx), multichannel = True), compssim(np.squeeze(targetx), np.squeeze(predout), multichannel = True)
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
    ax[0].imshow(X_test1_fig.reshape(target_height,target_width,1), cmap=plt.cm.gray, aspect='0.859375')
    ax[1].imshow(X_test_target1_fig.reshape(target_height,target_width,1), cmap=plt.cm.gray, aspect='0.859375')
    ax[2].imshow(preds1_fig.reshape(target_height,target_width,1), cmap=plt.cm.gray, aspect='0.859375')
    ax[0].axis('off')
    ax[0].set_title("Noisy test image", fontsize = 18)
    ax[1].axis('off')
    ax[1].set_title("Clear test image", fontsize = 18)
    ax[2].axis('off')
    ax[2].set_title("Denoised test image", fontsize = 18)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig('./neweval_ae_test_standard_%s.png'%(i))

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
