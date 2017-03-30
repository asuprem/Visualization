# Command level arguments
_BATCH = 32
_EPOCHS = 40
_CLASSES = 10
# Image augmentation may be neccesary for jitter and more examples
_AUGMENT = True

import pdb
# Import codes to obtain requisite commands
from keras.datasets import cifar10
from keras import callbacks as kCall
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils, generic_utils
from keras import backend as K
from keras.constraints import MaxNorm
K.set_image_dim_ordering('th')


# The plotting and math modules
import numpy as np
import matplotlib.pyplot as plt
import random



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
#Layer building
model = Sequential()

model.add(Convolution2D(32, (3,3),input_shape=(3,32,32),border_mode='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3,3), activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3,3),border_mode='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, (3,3), activation='relu',border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128, (3,3),border_mode='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, (3,3), activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(1024,activation='relu', W_constraint=MaxNorm(3)))
model.add(Dropout(.2))
model.add(Dense(512,activation='relu', W_constraint=MaxNorm(3)))
model.add(Dropout(.2))
model.add(Dense(_CLASSES, activation='softmax'))

#Compile with the adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.save("model_params")
#pdb.set_trace()

allModel = kCall.ModelCheckpoint("wt-{epoch:02d}.hdf5", monitor = 'val_loss', save_best_only=False, save_weights_only=True)
allCallbacks = [allModel]

modelHistory = model.fit(trainX,trainY,validation_data=(testX,testY), epochs=_EPOCHS, batch_size=_BATCH, callbacks=allCallbacks, verbose=2)
thefile = open('modelHistory.txt', 'w')
for item in modelHistory.history['val_loss']:
    thefile.write("%s\n" % item)
thefile.close()

scores = model.evaluate(testX, testY)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
model.save("finalModel")

