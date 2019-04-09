
# # Keras - the Happy House

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().magic('matplotlib inline')


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3 , name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name = 'fc')(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'HappyModel')
        
    return model


happyModel = HappyModel(X_train.shape[1:])
happyModel.compile(optimizer = "adam", loss='binary_crossentropy', metrics=['accuracy'])
happyModel.fit(x = X_train , y = Y_train, epochs = 40, batch_size = 16)
preds = happyModel.evaluate(x=X_test, y=Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

happyModel.summary()
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))


"""
Output:

number of training examples = 600
number of test examples = 150
X_train shape: (600, 64, 64, 3)
Y_train shape: (600, 1)
X_test shape: (150, 64, 64, 3)
Y_test shape: (150, 1)

Epoch 1/40
600/600 [==============================] - 15s - loss: 1.6394 - acc: 0.6900    
Epoch 2/40
600/600 [==============================] - 15s - loss: 0.3378 - acc: 0.8650    
Epoch 3/40
600/600 [==============================] - 15s - loss: 0.2169 - acc: 0.9217    
Epoch 4/40
600/600 [==============================] - 15s - loss: 0.1860 - acc: 0.9233    
Epoch 5/40
600/600 [==============================] - 15s - loss: 0.1007 - acc: 0.9567    
Epoch 6/40
600/600 [==============================] - 15s - loss: 0.1104 - acc: 0.9533    
Epoch 7/40
600/600 [==============================] - 15s - loss: 0.1190 - acc: 0.9650    
Epoch 8/40
600/600 [==============================] - 15s - loss: 0.1189 - acc: 0.9517    
Epoch 9/40
600/600 [==============================] - 15s - loss: 0.0773 - acc: 0.9750    
Epoch 10/40
600/600 [==============================] - 15s - loss: 0.0547 - acc: 0.9817    
Epoch 11/40
600/600 [==============================] - 15s - loss: 0.0507 - acc: 0.9883    
Epoch 12/40
600/600 [==============================] - 15s - loss: 0.0590 - acc: 0.9783    
Epoch 13/40
600/600 [==============================] - 15s - loss: 0.1253 - acc: 0.9567    
Epoch 14/40
600/600 [==============================] - 15s - loss: 0.0494 - acc: 0.9833    
Epoch 15/40
600/600 [==============================] - 15s - loss: 0.0928 - acc: 0.9700    
Epoch 16/40
600/600 [==============================] - 15s - loss: 0.0507 - acc: 0.9833    
Epoch 17/40
600/600 [==============================] - 15s - loss: 0.0671 - acc: 0.9817    
Epoch 18/40
600/600 [==============================] - 15s - loss: 0.1265 - acc: 0.9500    
Epoch 19/40
600/600 [==============================] - 15s - loss: 0.0444 - acc: 0.9817    
Epoch 20/40
600/600 [==============================] - 15s - loss: 0.0545 - acc: 0.9750    
Epoch 21/40
600/600 [==============================] - 15s - loss: 0.0383 - acc: 0.9867    
Epoch 22/40
600/600 [==============================] - 15s - loss: 0.0866 - acc: 0.9633    
Epoch 23/40
600/600 [==============================] - 15s - loss: 0.0439 - acc: 0.9867    
Epoch 24/40
600/600 [==============================] - 15s - loss: 0.0301 - acc: 0.9933    
Epoch 25/40
600/600 [==============================] - 15s - loss: 0.0910 - acc: 0.9750    
Epoch 26/40
600/600 [==============================] - 15s - loss: 0.2656 - acc: 0.9283    
Epoch 27/40
600/600 [==============================] - 15s - loss: 0.0723 - acc: 0.9800    
Epoch 28/40
600/600 [==============================] - 15s - loss: 0.0717 - acc: 0.9767    
Epoch 29/40
600/600 [==============================] - 15s - loss: 0.1959 - acc: 0.9400    
Epoch 30/40
600/600 [==============================] - 15s - loss: 0.0879 - acc: 0.9683    
Epoch 31/40
600/600 [==============================] - 15s - loss: 0.0595 - acc: 0.9817    
Epoch 32/40
600/600 [==============================] - 15s - loss: 0.1005 - acc: 0.9767    
Epoch 33/40
600/600 [==============================] - 15s - loss: 0.0376 - acc: 0.9867    
Epoch 34/40
600/600 [==============================] - 15s - loss: 0.0233 - acc: 0.9950    
Epoch 35/40
600/600 [==============================] - 15s - loss: 0.0355 - acc: 0.9867    
Epoch 36/40
600/600 [==============================] - 15s - loss: 0.0746 - acc: 0.9767    
Epoch 37/40
600/600 [==============================] - 15s - loss: 0.0308 - acc: 0.9900    
Epoch 38/40
600/600 [==============================] - 15s - loss: 0.0196 - acc: 0.9933    
Epoch 39/40
600/600 [==============================] - 15s - loss: 0.0207 - acc: 0.9917    
Epoch 40/40
600/600 [==============================] - 15s - loss: 0.0414 - acc: 0.9883


150/150 [==============================] - 2s     

Loss = 0.526406405767
Test Accuracy = 0.866666668256




_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
input_14 (InputLayer) (None, 64, 64, 3) 0
_________________________________________________________________
zero_padding2d_13 (ZeroPaddi (None, 70, 70, 3) 0
_________________________________________________________________
conv0 (Conv2D) (None, 64, 64, 32) 4736
_________________________________________________________________
bn0 (BatchNormalization) (None, 64, 64, 32) 128
_________________________________________________________________
activation_4 (Activation) (None, 64, 64, 32) 0
_________________________________________________________________
max_pool (MaxPooling2D) (None, 32, 32, 32) 0
_________________________________________________________________
flatten_3 (Flatten) (None, 32768) 0
9
_________________________________________________________________
fc (Dense) (None, 1) 32769
=================================================================
Total params: 37,633
Trainable params: 37,569
Non-trainable params: 64
_________________________________________________________________

"""
