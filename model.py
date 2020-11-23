#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:22:22 2019

@author: usuario
"""

import tensorflow 
import skimage as sk
from skimage import transform as tf
import os
import nibabel as nib
from scipy.ndimage.interpolation import rotate
import random
from os.path import join as pjoin
import tempfile
tmpdir=tempfile.mkdtemp()
from os.path import isfile
from sklearn.utils import shuffle
import numpy as np
import keras.models
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.models import*
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tempfile import TemporaryFile


def resize_img(img,size):
    img_resize=np.copy(img)
    factors=(
            float (size[0])/img.shape[0],
           float(size[1])/img.shape[1] ,
           1)
    return zoom(img_resize, factors, order=1)


size = (256,256)


data_path = '/Users/usuario/Desktop/estudios/HOSPITAL_LA_FE/ACTUAL/CASOS_ALZHEIMER_EECC'
files = os.listdir(data_path)

p=0
for f in files:
   
    print(f)
    if f != '.DS_Store':
        path = os.path.join(data_path, f)
        subcarpetas=os.listdir(path)
        for a in subcarpetas:
            print(a)
            if a !='.DS_Store':
                 path_flair=os.path.join(path, a,'flair.nii')
                 path_lesion=os.path.join(path,a,'lesion.nii')
                 imgF = nib.load(path_flair)
                 imgF = imgF.get_data()
                 imgF_res = resize_img(imgF, size)
                
                 if p==0:
                    img_all_flair=imgF_res
                 else:
                    
                 
                    img_all_flair = np.concatenate( (img_all_flair, imgF_res),axis=2)
                    
               
                 
                
                 imgL = nib.load(path_lesion)
                 imgL =imgL.get_data()
                 imgL_res= resize_img(imgL, size)
                
                 
                 if p==0:
                    img_all_lesion=imgL_res
                 else:
                    
                 
                    img_all_lesion = np.concatenate( (img_all_lesion, imgL_res),axis=2)
                    
                 
                
                 p=p+1
                 
                
           
                
   
img_all_flair=np.rollaxis(img_all_flair,2,0)
img_all_lesion=np.fliplr(img_all_lesion)
img_all_lesion=np.rollaxis(img_all_lesion,2,0) 
img_all_flair=np.reshape(img_all_flair,(img_all_flair.shape[0],img_all_flair.shape[1],img_all_flair.shape[2],1))

img_all_lesion=np.reshape(img_all_lesion,(img_all_lesion.shape[0],img_all_lesion.shape[1],img_all_lesion.shape[2],1))


smooth=1



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def get_unet(input_shape=(256,256,1)):
    inputs = Input(shape=input_shape)
    
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    batch1 = BatchNormalization()(conv1)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(batch1)
    batch11 = BatchNormalization()(conv11)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch11)

    

    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    batch2 = BatchNormalization()(conv2)
    conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch2)
    batch22 = BatchNormalization()(conv22)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch22)
    

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    batch3= BatchNormalization()(conv3)
    conv33 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch3)
    batch33= BatchNormalization()(conv33)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch33)
    

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    batch4 = BatchNormalization()(conv4)
    conv44 = Conv2D(256, (3, 3), activation='relu', padding='same')(batch4)
    batch44 = BatchNormalization()(conv44)
    #drop4 = Dropout(0.5)(batch44)
    pool4 = MaxPooling2D(pool_size=(2, 2))(batch44)
    
    

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    batch5 = BatchNormalization()(conv5)
    conv55 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch5)
    batch55 = BatchNormalization()(conv55)
    #drop5 = Dropout(0.5)(batch55)
    
    up6=concatenate([UpSampling2D(size=(2,2))(batch55),batch44])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    batch6 = BatchNormalization()(conv6)
    conv66 = Conv2D(256, (3, 3), activation='relu', padding='same')(batch6)
    batch66 = BatchNormalization()(conv66)
    
    up7=concatenate([UpSampling2D(size=(2,2))(batch66),batch33])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    batch7 = BatchNormalization()(conv7)
    conv77 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch7)
    batch77 = BatchNormalization()(conv77)
    
    up8=concatenate([UpSampling2D(size=(2,2))(batch77),batch22])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    batch8 = BatchNormalization()(conv8)
    conv88 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch8)
    batch88 = BatchNormalization()(conv88)
    
    
    up9=concatenate([UpSampling2D(size=(2,2))(batch88),batch11])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    batch9 = BatchNormalization()(conv9)
    conv99 = Conv2D(32, (3, 3), activation='relu', padding='same')(batch9)
    batch99 = BatchNormalization()(conv99)


    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(batch99)
    

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model



img_all_flair1=img_all_flair
img_all_lesion1=img_all_lesion

img_all_lesion1=np.flip(img_all_lesion1,axis=1) 
img_all_flair1=np.flip(img_all_flair1,axis=1) 
noise_variance=(0, 1)

def augment_gaussian_noise(data_sample ,noise_variance):
    
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample

img_all_flair2=augment_gaussian_noise(img_all_flair,noise_variance)

img_all_lesion2=img_all_lesion


img_all_lesionT = np.concatenate( (img_all_lesion, img_all_lesion1),axis=0)
img_all_lesionT= np.concatenate( (img_all_lesionT, img_all_lesion2),axis=0)


img_all_flairT = np.concatenate( (img_all_flair, img_all_flair1),axis=0)
img_all_flairT= np.concatenate( (img_all_flairT, img_all_flair2),axis=0)

indice=np.arange(8685)
np.random.shuffle(indice)

img_all_flairT = img_all_flairT[indice]
img_all_lesionT = img_all_lesionT[indice]

def data_train():
    
 
 imgs_f=img_all_flairT  

 mean = np.mean(imgs_f)
 
 desv=np.std(imgs_f)
 
 imgs_f -= mean
 
 imgs_f /= desv
 
 

    
 imgs_train = imgs_f[1:6948,:,:,:]
 imgs_test=imgs_f[6948:8685,:,:,:]
 

    
 imgs_l=img_all_lesionT
 


 imgs_mask_train = imgs_l[1:6948,:,:,:]
 
    
    
 model= get_unet()
 model.summary()
   
 model_checkpoint = ModelCheckpoint('pesosAUMENTADO21.h5', monitor='val_loss', save_best_only=True)


 model.fit(imgs_train, imgs_mask_train, batch_size=20, epochs=10, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    
    
 imgs_mask_test = model.predict(imgs_test, verbose=1)
 np.save('imgs_mask_testAUMENTADO21.npy', imgs_mask_test)
