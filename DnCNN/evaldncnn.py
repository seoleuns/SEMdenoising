## For the evaluation of DnCNN
##
## Seoleun Shin (KRISS), 2023-June
#
## Use metrics based on https://github.com/up42/image-similarity-measures
#  and modified for our samples (Seoleun Shin, 2023/June/01) ##
##---------------------------------------------------------------------------------------------------------
##
import argparse
import logging
import os, time, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
#from keras import backend as K
#import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
#from skimage.measure import peak_signal_noise_ratio, compare_ssim
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as compssim
import models
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
from image_similarity_measures.evaluate import evaluation

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
#parser.add_argument('--train_data', default='./data/npy_data/clean_patches.npy', type=str, help='path of train data')
#parser.add_argument('--test_dir', default='./data/Test/Set68', type=str, help='directory of test dataset')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=25, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
parser.add_argument('--pretrain', default=None, type=str, help='path of pre-trained model')
parser.add_argument('--only_test', default=False, type=bool, help='train and test or only test')
args = parser.parse_args()

target_height = 512
target_width = 512
target_height = 440
target_width = 440
image_size =440

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
            #noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
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
    #ckpt = ModelCheckpoint(save_dir+'/model_{epoch:02d}.h5', monitor='val_loss', 
    ckpt = ModelCheckpoint(save_dir+'/model_may_12.h5', monitor='val_loss', 
                    verbose=0, period=args.save_every)
    csv_logger = CSVLogger(save_dir+'/log.csv', append=True, separator=',')
    lr = LearningRateScheduler(step_decay)
    # train 
    history = model.fit_generator(train_datagen(data, batch_size=args.batch_size),
                    steps_per_epoch=len(data)//args.batch_size, epochs=args.epoch, verbose=1, 
                    callbacks=[ckpt, csv_logger, lr])
    
    return model

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

def renormal_load_image_from_dir3(img_path):
    file_list = glob.glob(img_path+'/*.tif')
    #file_list = glob.glob(img_path)
    file_list.sort()
    img_list = np.empty((len(file_list), target_height, target_width, 1))
    for i, fig in enumerate(file_list):
        figures = cv2.imread(fig, cv2.IMREAD_GRAYSCALE)
        figures = cropImage(figures,(0,0),(512,400))
        img = tf.keras.utils.img_to_array(figures).astype('float32')
        img_array = tf.image.resize(img, size=[target_height, target_width], antialias=True)
        img_list[i] = tf.clip_by_value(img_array/255.0, 0.0, 1.0)
    return img_list

def renormal_load_image_from_dir4(img_path):
    file_list = glob.glob(img_path+'/*.tif')
    file_list.sort()
    #file_list = glob.glob(img_path)
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



def test(model):
    
    print('Start to test on {}'.format(args.test_dir))
    out_dir = save_dir + args.test_dir.split('/')[-1] + '/'
    if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            
    name = []
    psnrsk = []
    ssim = []
    rmse = []
    fsim = []
    issm = []
    uiq = []
    sam = []
    psnrcheck = []
    #file_list = glob.glob('{}/*.jpg'.format(args.test_dir))
    file_list = renormal_load_image_from_dir4('./data/Clean')
    file_list2 = renormal_load_image_from_dir4('./data/Noisy')

    i = 0
    for file, file2 in zip(file_list, file_list2):
        # read image
        fileclean = file
        img_test = file2
        # predict
        X_test1_fig = img_test 
        x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1) 
        
        y_predict = model.predict(x_test)
        # calculate numeric metrics
        img_out = y_predict.reshape(file.shape)
        preds1 = img_out
        img_testcheck = x_test.reshape(file.shape)
        X_test_target1 = fileclean
        img_out = np.clip(img_out, 0, 1)
        psnr_noise, psnr_denoised = peak_signal_noise_ratio(fileclean, img_testcheck), peak_signal_noise_ratio(fileclean, img_out)
        ssim_noise, ssim_denoised = compssim(np.squeeze(fileclean), np.squeeze(img_testcheck)), compssim(np.squeeze(fileclean), np.squeeze(img_out))
        rmse_noise, rmse_denoised = evaluation(fileclean, img_testcheck, metrics=["rmse"]), evaluation(fileclean, img_out, metrics=["rmse"])
        fsim_noise, fsim_denoised = evaluation(fileclean, img_testcheck, metrics=["fsim"]), evaluation(fileclean, img_out, metrics=["fsim"])
        uiq_noise, uiq_denoised = evaluation(fileclean, img_testcheck, metrics=["uiq"]), evaluation(fileclean, img_out, metrics=["uiq"])
        psnrcheck_noise, psnrcheck_denoised = evaluation(fileclean, img_testcheck, metrics=["psnr"]), evaluation(fileclean, img_out, metrics=["psnr"])
        psnrsk.append(psnr_denoised)
        psnrcheck.append(psnr_denoised)
        ssim.append(ssim_denoised)
        rmse.append(rmse_denoised)
        fsim.append(fsim_denoised)
        uiq.append(uiq_denoised)

        i = i+1

        rows=1
        cols=3
        fig, ax = plt.subplots(1,3,figsize=(16,5), facecolor='white')
        fig.tight_layout()
        fig.set_facecolor("white")
        ax[0].imshow(X_test1_fig.reshape(image_size,image_size), cmap=plt.cm.gray, aspect='0.78125')
        ax[1].imshow(X_test_target1.reshape(image_size,image_size), cmap=plt.cm.gray, aspect='0.78125')
        ax[2].imshow(preds1.reshape(image_size,image_size), cmap=plt.cm.gray, aspect='0.78125')
        ax[0].axis('off')
        ax[0].set_title("Noisy test image", fontsize = 18)
        ax[1].axis('off')
        ax[1].set_title("Clear test image", fontsize = 18)
        ax[2].axis('off')
        ax[2].set_title("Denoised test image", fontsize = 18)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        #fig.savefig('./neweval_dncnn_test_standard_%s.png'%(i), dpi=100, bbox_inches='tight', pad_inches = 0)
        fig.savefig('./neweval_dncnn_test_standard_%s.png'%(i))
        plt.close()
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
    
if __name__ == '__main__':   
    
    if args.only_test:
        model = load_model(args.pretrain, compile=False)
        test(model)
    else:
        model = train()
        test(model)       
    
