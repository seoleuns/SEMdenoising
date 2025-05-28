## Code is based on https://github.com/husqin/DnCNN-keras and
## Modified for SEM-Denoising by Seoleun Shin (KRISS, 2023-05-01)
##
## Please follow the instruction in https://github.com/husqin/DnCNN-keras
##
#
import glob
import os
import cv2
import numpy as np
from multiprocessing import Pool

from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
#from skimage.measure import peak_signal_noise_ratio, compare_ssim
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compssim
import models
import cv2
import tensorflow as tf

patch_size, stride = 40, 10
aug_times = 1
target_height = 440
target_width = 440
image_size = 440

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
        figures = cropImage(figures,(0,0),(440,440))
        height = tf.shape(figures)[0]
        width = tf.shape(figures)[1]
        crop_size = tf.minimum(height, width)
        img_array = tf.keras.utils.img_to_array(figures).astype('float32')
        img_list[i] = tf.clip_by_value(img_array/255.0, 0.0, 1.0)
    return img_list

def load_image_from_dir(img_path):
    file_list = glob.glob(img_path+'/*.tif')
    file_list.sort()
    return file_list


def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):

    scales = [1, 0.9, 0.8, 0.7]
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img = cropImage(img,(0,0),(440,440))
    h, w = img.shape
    crop_size = tf.minimum(h, w)
    patches = []

    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)

        img_scaled = cv2.resize(img, (h_scaled, w_scaled),
                                interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    return patches


if __name__ == '__main__':
    # parameters
    save_dir = 'Data/npy_data/'
    file_list = load_image_from_dir('../Data/Time/low')
    num_threads = 16
    num_threads = 1
    nfile1=len(file_list)
    print('Start...', nfile1)
    # initrialize
    res = []
    # generate patches
    for i in range(0, nfile1, num_threads):
        patch=gen_patches(file_list[i])
        for x in patch:
            res.append(x)

        print('Picture '+str(i)+' to '+str(i+num_threads)+' are finished...')

    # save to .npy
    res = np.array(res, dtype='uint8')
    print('Shape of result = ' + str(res.shape))
    print('Saving data...')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(save_dir+'semclean_patches.npy', res)
    print('Done.')


# In[10]:


import numpy as np
from keras.models import *
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract

def DnCNN():
    
    inpt = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model



import argparse
import logging
import os, time, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import models
from importlib import reload

## Params
class args:
    only_test=False
    pretrain=None
    save_every=5
    lr=1e-3
    epoch=50
    sigma=25
    test_dir='Data/test_cleaned'
    train_data='Data/npy_data/semclean_patches.npy'
    batch_size=128
    model='DnCNN'
    
if False:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--train_data', default='Data/npy_data/semclean_patches.npy', type=str, help='path of train data')
    parser.add_argument('--test_dir', default='Data/test_cleaned', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
    parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
    parser.add_argument('--pretrain', default=None, type=str, help='path of pre-trained model')
    parser.add_argument('--only_test', default=False, type=bool, help='train and test or only test')
    args = parser.parse_args()

if not args.only_test:
    save_dir = './snapshot/sem/save_'+ args.model + '_' + 'sigma' + str(args.sigma) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # log
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S',
                    filename=save_dir+'info.log',
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(args)
    
else:
    save_dir = '/'.join(args.pretrain.split('/')[:-1]) + '/'

      
def load_train_data():
    
    logging.info('loading train data...')   
    data = np.load(args.train_data)
    logging.info('Size of train data: ({}, {}, {})'.format(data.shape[0],data.shape[1],data.shape[2]))
    
    return data

def step_decay(epoch):
    
    initial_lr = args.lr
    if epoch<50:
        lr = initial_lr
    else:
        lr = initial_lr/10
    
    return lr

def train_datagen(y_, batch_size=8):
    
    # y_ is the tensor of clean patches
    indices = list(range(y_.shape[0]))
    while(True):
        np.random.shuffle(indices)    # shuffle
        for i in range(0, len(indices), batch_size):
            ge_batch_y = y_[indices[i:i+batch_size]]
            noise =  np.random.normal(0, args.sigma/255.0, ge_batch_y.shape)    # noise
            ge_batch_x = ge_batch_y + noise  # input image = clean image + noise
            yield ge_batch_x, ge_batch_y
        
def train():
    
    data = load_train_data()
    data = data.reshape((data.shape[0],data.shape[1],data.shape[2],1))
    data = data.astype('float32')/255.0
    # model selection
    if args.pretrain:   model = load_model(args.pretrain, compile=False)
    else:   
        if args.model == 'DnCNN': model = models.DnCNN()
    # compile the model
    model.compile(optimizer=Adam(), loss=['mse'])
    
    # use call back functions
    ckpt = ModelCheckpoint(save_dir+'/model_{epoch:02d}.h5', monitor='val_loss', 
                    verbose=0, save_freq=args.save_every)
    csv_logger = CSVLogger(save_dir+'/log.csv', append=True, separator=',')
    lr = LearningRateScheduler(step_decay)
    # train 
    history = model.fit_generator(train_datagen(data, batch_size=args.batch_size),
                    steps_per_epoch=len(data)//args.batch_size, epochs=args.epoch, verbose=1, 
                    callbacks=[ckpt, csv_logger, lr])
    
    return model

def test(model):
    
    print('Start to test on {}'.format(args.test_dir))
    out_dir = save_dir + args.test_dir.split('/')[-1] + '/'
    if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            
    name = []
    psnr = []
    ssim = []
    import glob
    file_list=[]
    filelist = renormal_load_image_from_dir('../Data/Line/Clean')
    filelist2 = renormal_load_image_from_dir('../Data/Line/Noisy')
    for file in filelist:
        file=file.replace("\\", "/")
        file_list.append(file)
    for file in file_list:
        print(file)
        # read image
        img_clean = file
        img_test = img_clean + np.random.normal(0, args.sigma/255.0, img_clean.shape)
        img_test = img_test.astype('float32')
        # predict
        x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1) 
        y_predict = model.predict(x_test)
        img_testcheck = x_test.reshape(file.shape)
        img_testcheck = np.squeeze(img_testcheck)
        # calculate numeric metrics
        img_out = y_predict.reshape(img_clean.shape)
        img_out = np.clip(img_out, 0, 1)
        img_out = np.squeeze(img_out)
        fileclean = np.squeeze(file)

        psnr_noise, psnr_denoised = peak_signal_noise_ratio(fileclean, img_testcheck), peak_signal_noise_ratio(fileclean, img_out)
        ssim_noise, ssim_denoised = compssim(fileclean, img_testcheck), compssim(fileclean, img_out)
        psnr.append(psnr_denoised)
        ssim.append(ssim_denoised)
        # save images
        filename = str(i)
        print(filename)
        name.append(filename)
        img_testcheck = Image.fromarray((img_testcheck*255).astype('uint8'))
        img_testcheck.save(out_dir+filename+'_sigma'+'{}_psnr{:.2f}.png'.format(args.sigma, psnr_noise))
        img_out = Image.fromarray((img_out*255).astype('uint8'))
        img_out.save(out_dir+filename+'_psnr{:.2f}.png'.format(psnr_denoised))

    
    psnr_avg = sum(psnr)/len(psnr)
    ssim_avg = sum(ssim)/len(ssim)
    name.append('Average')
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))
    
    pd.DataFrame({'name':np.array(name), 'psnr':np.array(psnr), 'ssim':np.array(ssim)}).to_csv(out_dir+'/metrics.csv', index=True)
    
if __name__ == '__main__':   
    
    if args.only_test:
        model = load_model(args.pretrain, compile=False)
        test(model)
    else:
        model = train()
        test(model)       
    
