
# coding: utf-8

# # Keras - the Happy House

# In[1]:

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


# In[2]:

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


# In[15]:


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)
    #X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64, (3, 3), strides = (1, 1), padding='same', name = 'z1')(X_input)
    X = BatchNormalization(axis = 3 , name = 'bn1')(X)
    X = Activation('relu')(X)
    X = Conv2D(64, (3, 3), strides = (1, 1), padding='same', name = 'z2')(X)
    X = BatchNormalization(axis = 3 , name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool1')(X)
    X = Conv2D(128, (3, 3), strides = (1, 1), padding='same', name = 'z3')(X)
    X = BatchNormalization(axis = 3 , name = 'bn3')(X)
    X = Activation('relu')(X)
    X = Conv2D(128, (3, 3), strides = (1, 1), padding='same', name = 'z4')(X)
    X = BatchNormalization(axis = 3 , name = 'bn4')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool2')(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), padding='same', name = 'z5')(X)
    X = BatchNormalization(axis = 3 , name = 'bn5')(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), padding='same', name = 'z6')(X)
    X = BatchNormalization(axis = 3 , name = 'bn6')(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), padding='same', name = 'z7')(X)
    X = BatchNormalization(axis = 3 , name = 'bn7')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool3')(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), padding='same', name = 'z8')(X)
    X = BatchNormalization(axis = 3 , name = 'bn8')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), padding='same', name = 'z9')(X)
    X = BatchNormalization(axis = 3 , name = 'bn9')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), padding='same', name = 'z10')(X)
    X = BatchNormalization(axis = 3 , name = 'bn10')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool4')(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), padding='same', name = 'z11')(X)
    X = BatchNormalization(axis = 3 , name = 'bn11')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), padding='same', name = 'z12')(X)
    X = BatchNormalization(axis = 3 , name = 'bn12')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), padding='same', name = 'z13')(X)
    X = BatchNormalization(axis = 3 , name = 'bn13')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool5')(X)
    X = Flatten()(X)
    X = Dense(4096, activation='sigmoid', name = 'fc1')(X)
    X = Dense(4096, activation='sigmoid', name = 'fc2')(X)
    X = Dense(1, activation='sigmoid', name = 'fc3')(X)

    model = Model(inputs = X_input, outputs = X, name = 'HappyModel')
    
    
    return model


# In[18]:

happyModel = HappyModel(X_train.shape[1:])
#print(X_train.shape[1:])


# In[19]:

happyModel.compile(optimizer = "adam", loss='binary_crossentropy', metrics=['accuracy'])


# In[20]:

happyModel.fit(x = X_train , y = Y_train, epochs = 2, batch_size = 16)


# In[21]:

preds = happyModel.evaluate(x=X_test, y=Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[22]:

img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))


# In[23]:

happyModel.summary()


# In[24]:

plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))


# In[ ]:



