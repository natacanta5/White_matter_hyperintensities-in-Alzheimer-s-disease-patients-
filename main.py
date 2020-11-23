#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:47:47 2019

@author: usuario
"""

import os
import nibabel as nib
from os.path import join as pjoin
import tempfile
tmpdir=tempfile.mkdtemp()
from os.path import isfile
from sklearn.utils import shuffle
import numpy as np
import keras.models
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
outfile = TemporaryFile()
outfile1=TemporaryFile()

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
                 imgF_res=np.reshape(imgF_res,(imgF_res.shape[2],size[0],size[1],1))
                 
                 if p==0:
                    img_all_flair=imgF_res
                 else:
                    
                 
                    img_all_flair = np.concatenate( (img_all_flair, imgF_res),axis=0)
                    
               
                 #np.save(os.path.join('/Volumes/Natalia/NUEVO_TFG', 'train_setFLAIR'), img_all_flair)
                 imgL = nib.load(path_lesion)
                 imgL =imgL.get_data()
                 imgL_res= resize_img(imgL, size)
                 
                 imgL_res=np.reshape(imgL_res,(imgL_res.shape[2],size[0],size[1],1))
                 
                 if p==0:
                    img_all_lesion=imgL_res
                 else:
                    
                 
                    img_all_lesion = np.concatenate( (img_all_lesion, imgL_res),axis=0)
                    
                 p=p+1
                 
                # np.save(os.path.join('/Volumes/Natalia/NUEVO_TFG', 'train_setLESION'), img_all_lesion)



smooth=1

y_true=img_all_flair
y_pred=img_all_lesion

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
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model




def data_train():
    
    imgs_train = y_true.astype('float64')
    mean = np.mean(imgs_train) 

    imgs_train -= mean
    imgs_train /= np.std(imgs_train) 

    imgs_mask_train = y_pred.astype('uint16')
    imgs_mask_train =imgs_mask_train/255
    
    imgs_test = np.load('/Volumes/Natalia/NUEVO_TFG/train_setFLAIR.npy')
    
    model= get_unet()
    model.summary()

   
    model_checkpoint = ModelCheckpoint('peso.h5', monitor='val_loss', save_best_only=True)


    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=40, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    
    #model.load_weights('weights.h5')
    
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test1.npy', imgs_mask_test)