
from model import *

model=get_unet()
model.load_weights('pesos2.h5')

imgs_mi=img_all_flairT 

mean1 =np.mean(imgs_mi)

desv1 =np.std(imgs_mi)

T =nib.load('/Users/usuario/Desktop/flair1.nii')
T1=T.get_data()
T2 =resize_img(T1, size)

T3=np.rollaxis(T2,2,0)

T4=np.reshape(T3,(T3.shape[0],T3.shape[1],T3.shape[2],1))




T7= (T4-mean1)/desv1


preds_train = model.predict(T7, verbose=1)

np.save("/Users/usuario/Desktop/prueba21",preds_train)

de=np.load('prueba21.npy')

de[de[...,0] > 0.5] = 1      #thresholding 
de[de[...,0] <= 0.5] = 0