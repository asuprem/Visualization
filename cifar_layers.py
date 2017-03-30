_MODEL = 39
_BATCH = 32
_EPOCHS = 40
_CLASSES = 10
_BASEDIM = 512
_FIGDIM = 150
_VISITERS = 50
# Image augmentation may be neccesary for jitter and more examples
_AUGMENT = True

import matplotlib
matplotlib.use('Agg')

import keras
from keras.datasets import cifar10
from keras import callbacks as kCall
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.engine.topology import InputLayer
from keras.utils import np_utils, generic_utils
from keras import backend as K
from keras.constraints import MaxNorm
from keras import layers
K.set_image_dim_ordering('th')

import numpy as np
import matplotlib.pyplot as plt
import random, pdb, math, os
from PIL import Image
from tqdm import tqdm
from tensor_utils import tensor_to_image

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
scores=model.evaluate(testX[0:nums,:,:,:],testY[0:nums,:])
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

#Create model up to the flatten layer (i.e. keeping only the convolutional layers)
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

#----------------------
#----------------------
#----------------------
folderPath  = "wt-"+str(_MODEL).zfill(2)
if not os.path.exists(folderPath):
        os.makedirs(folderPath)
for layerSelect in tqdm(convL):
    baseDim = _BASEDIM
    in_image = vis_model.input
    nu = 1
    iterations = _VISITERS
    baseForm = []
    assert(vis_model.layers[layerSelect].rank == 2)
    layer_dict = dict([(layer.name, layer) for layer in vis_model.layers])
    layer_name = vis_model.layers[layerSelect].name
    nFilters = vis_model.layers[layerSelect].get_config()['filters']
        
    for filterSelect in tqdm(range(nFilters)): 
        baseImage = np.random.random((1, 3,baseDim,baseDim))
        baseImage = 20*(baseImage-0.2) + 128
        layerOut = layer_dict[layer_name].output
        loss = K.mean(layerOut[:,filterSelect,:,:])
        gradients = K.gradients(loss,in_image)[0]
        gradients /= K.sqrt(K.mean(K.square(gradients))) + 1e-6
        lossGrad = K.function([in_image, K.learning_phase()], [loss, gradients])

        for i in range(iterations):
            lossN, gradN = lossGrad([baseImage,1])
            #pdb.set_trace()
            baseImage += gradN*nu
        baseForm.append(np.copy(baseImage))

    fig = plt.figure(figsize=(_FIGDIM,_FIGDIM))
    side=math.ceil(math.sqrt(nFilters))
    convPath = "conv-layer"+str(layerSelect)
    if not os.path.exists(os.path.join(folderPath,convPath)):
        os.makedirs(os.path.join(folderPath,convPath))
    for i in tqdm(range(nFilters)):
        fileName = "filter-" + str(i).zfill(3) + ".png"
        plt.imshow(tensor_to_image(baseForm[i][0]))
        plt.savefig(os.path.join(folderPath,convPath,fileName),bbox_inches='tight')


