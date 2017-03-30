_MODEL = 39
_BATCH = 32
_EPOCHS = 40
_CLASSES = 10
# Image augmentation may be neccesary for jitter and more examples
_AUGMENT = True

import matplotlib
matplotlib.use('Agg')

from keras.datasets import cifar10
from keras import callbacks as kCall
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.engine.topology import InputLayer
from keras.utils import np_utils, generic_utils
from keras import backend as K
from keras.constraints import MaxNorm
K.set_image_dim_ordering('th')
import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb
from keras import layers
import math
from tqdm import tqdm
from tensor_utils import tensor_to_image
import os

(trainX,trainY),(testX,testY) = cifar10.load_data()
print ('train Shape:  ', trainX.shape)
print ('train Exmpl:  ', trainX.shape[0])
print ('test  Exmpl:  ', testX.shape[0])
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0

trainY  = np_utils.to_categorical(trainY)
testY   = np_utils.to_categorical(testY)
_CLASSES = trainY.shape[1]

wtN = _MODEL
wtN = str(wtN).zfill(2)
file="weights/wt-" + wtN + ".hdf5"
model = keras.models.Sequential()
model = keras.models.load_model("model_params")
model.load_weights(file)

nums=500
scores=model.evaluate(testX,testY)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

layerCount=-1
for layer in model.layers:
    layerCount+=1
    if layer.__class__.__name__ == "Conv2D":
        print (str(layerCount).rjust(2) + "  " + 
               layer.name.ljust(15) + "  :  " + 
               layer.get_config()['name'].ljust(15) + "  " + 
               str(layer.get_config()['filters']) + "  " + 
               str(layer.get_config()['kernel_size']))
    elif layer.__class__.__name__ == "Flatten":
        fLayer = layerCount  
    else:
        print (str(layerCount).rjust(2) + "  " + layer.__class__.__name__.ljust(15) + "  :  " + layer.get_config()['name'])

layerOutput = fLayer-1
vis_model = Sequential()
vis_model.add(InputLayer(input_shape=(3,None,None)))
cLayer=0
for layer in model.layers:
    nLayer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': layer.get_config()})
    vis_model.add(nLayer)
    nLayer.set_weights(layer.get_weights())
    cLayer+=1
    if cLayer>layerOutput:
        break
print (vis_model.summary())

layerCount=-1
convL = []
for layer in vis_model.layers:
    layerCount+=1
    if layer.__class__.__name__ == "Conv2D":
        print (str(layerCount).rjust(2) + "  " + 
               layer.name.ljust(15) + "  :  " + 
               layer.get_config()['name'].ljust(15) + "  " + 
               str(layer.get_config()['filters']) + "  " + 
               str(layer.get_config()['kernel_size']))
        convL.append(layerCount)
    else:
        print (str(layerCount).rjust(2) + "  " + layer.name.ljust(15) + "  :  " + layer.get_config()['name'])


max_activations = [np.zeros(vis_model.layers[convLayer].get_config()['filters']) for convLayer in convL]
max_image_index = [np.zeros(vis_model.layers[convLayer].get_config()['filters']) for convLayer in convL]
vis_input = vis_model.input
loss = [K.mean(layer.output,axis=[2,3]) for layer in [vis_model.layers[i] for i in convL]]
#loss = [K.mean(layer.output,axis=[2,3]) for layer in vis_model.layers]
lossGrad = K.function([vis_input, K.learning_phase()], loss)
for img in tqdm(range(60000)):
    lossT = lossGrad([trainX[img:img+1],0])
    residues=[max_activations[i]-lossT[i] for i in range(len(convL))]
    for i in range(len(convL)):
        max_image_index[i][np.where(residues[i]<0)[1]] = img
        max_activations[i][np.where(residues[i]<0)[1]]=lossT[i][0,np.where(residues[i]<0)[1]]
name_lst = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
folderPath = "activations"
if not os.path.exists(folderPath):
        os.makedirs(folderPath)
layerCount = 0
for layer in max_image_index:
    filtCount = 0
    for image_index in layer:
        fileName = "convlayer-"+str(convL[layerCount]).zfill(2)+"_filter-"+str(filtCount).zfill(3)+"_"+name_lst[int(np.dot(np.linspace(0,9,10),testY[image_index]))]+".png"
        Image.fromarray(trainX[image_index,:,:,:].transpose(1,2,0)).save(os.path.join(folderPath,fileName))
        filtCount+=1
    layerCount+=1
