
# coding: utf-8

# # AlexNet 
# 

# ## TensorFlow model
# 
# Following is an VGG-16 model implementation with some of the hyperparameter changes designed for SIGNS dataset.
# 
# VggNet model implementation is given in paper K. Simonyan, A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. Proceedings of ICLR 2015, p.1-14 
# 
# VggNet is the winner of the 2014 edition of the ImageNet classification challenge (ILSVRC-2014).
# 

# In[1]:

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

get_ipython().magic('matplotlib inline')
np.random.seed(1)


# Run the next cell to load the "SIGNS" dataset you are going to use.

# In[2]:

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


# In[3]:

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


# In[4]:

# GRADED FUNCTION: create_placeholders

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(tf.float32 , shape = (None, n_H0, n_W0, n_C0) )
    Y = tf.placeholder(tf.float32 , shape = (None, n_y) )
    ### END CODE HERE ###
    
    return X, Y


# In[5]:

X, Y = create_placeholders(64, 64, 3, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))


# **Expected Output**
# 
# <table> 
# <tr>
# <td>
#     X = Tensor("Placeholder:0", shape=(?, 64, 64, 3), dtype=float32)
# 
# </td>
# </tr>
# <tr>
# <td>
#     Y = Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)
# 
# </td>
# </tr>
# </table>

# In[6]:

# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable("W1", [3,3,3,64], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [3,3,64,64], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3", [3,3,64,128], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W4", [3,3,128,128], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W5 = tf.get_variable("W5", [3,3,128,256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W6 = tf.get_variable("W6", [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W7 = tf.get_variable("W7", [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W8 = tf.get_variable("W8", [3,3,256,512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W9 = tf.get_variable("W9", [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W10 = tf.get_variable("W10", [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W11 = tf.get_variable("W11", [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W12 = tf.get_variable("W12", [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W13 = tf.get_variable("W13", [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))


    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "W5": W5,
                  "W6": W6,
                  "W7": W7,
                  "W8": W8,
                  "W9": W9,
                  "W10": W10,
                  "W11": W11,
                  "W12": W12,
                  "W13": W13}
    
    return parameters


# In[7]:

tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
    print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
    print("W3 = " + str(parameters["W3"].eval()[1,1,1]))
    print("W4 = " + str(parameters["W4"].eval()[1,1,1]))
    print("W5 = " + str(parameters["W5"].eval()[1,1,1]))
    print("W6 = " + str(parameters["W6"].eval()[1,1,1]))
    print("W7 = " + str(parameters["W7"].eval()[1,1,1]))
    print("W8 = " + str(parameters["W8"].eval()[1,1,1]))
    print("W9 = " + str(parameters["W9"].eval()[1,1,1]))
    print("W10 = " + str(parameters["W10"].eval()[1,1,1]))
    print("W11 = " + str(parameters["W11"].eval()[1,1,1]))
    print("W12 = " + str(parameters["W12"].eval()[1,1,1]))
    print("W13 = " + str(parameters["W13"].eval()[1,1,1]))


# ** Expected Output:**
# 
# <table> 
# 
#     <tr>
#         <td>
#         W1 = 
#         </td>
#         <td>
# [ 0.00131723  0.14176141 -0.04434952  0.09197326  0.14984085 -0.03514394 <br>
#  -0.06847463  0.05245192]
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#         W2 = 
#         </td>
#         <td>
# [-0.08566415  0.17750949  0.11974221  0.16773748 -0.0830943  -0.08058 <br>
#  -0.00577033 -0.14643836  0.24162132 -0.05857408 -0.19055021  0.1345228 <br>
#  -0.22779644 -0.1601823  -0.16117483 -0.10286498]
#         </td>
#     </tr>
# 
# </table>

# In[8]:

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
    W6 = parameters['W6']
    W7 = parameters['W7']
    W8 = parameters['W8']
    W9 = parameters['W9']
    W10 = parameters['W10']
    W11 = parameters['W11']
    W12 = parameters['W12']
    W13 = parameters['W13']
    
    ### START CODE HERE ###
    # CONV2D: filters W1, window 3x3, stride 1, padding 'SAME'
    Z1 = tf.nn.conv2d( X, W1, strides = [1,1,1,1], padding = 'SAME')
    # ?? have not taken b1 parameter
    # RELU
    A1 = tf.nn.relu(Z1)
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(A1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 2x2, sride 2, padding 'VALID'
    P1 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # CONV2D: filters W3, stride 1, padding 'SAME'
    Z3 = tf.nn.conv2d(P1,W3, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A3 = tf.nn.relu(Z3)
    # CONV2D: filters W4, stride 1, padding 'SAME'
    Z4 = tf.nn.conv2d(A3,W4, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A4 = tf.nn.relu(Z4)
    # MAXPOOL: window 2x2, sride 2, padding 'VALID'
    P2 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # CONV2D: filters W5, stride 1, padding 'SAME'
    Z5 = tf.nn.conv2d(P2,W5, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A5 = tf.nn.relu(Z5)
    # CONV2D: filters W6, stride 1, padding 'SAME'
    Z6 = tf.nn.conv2d(A5,W6, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A6 = tf.nn.relu(Z6)
    # CONV2D: filters W7, stride 1, padding 'SAME'
    Z7 = tf.nn.conv2d(A6,W7, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A7 = tf.nn.relu(Z7)
    # MAXPOOL: window 2x2, sride 2, padding 'VALID'
    P3 = tf.nn.max_pool(A7, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # CONV2D: filters W8, stride 1, padding 'SAME'
    Z8 = tf.nn.conv2d(P3,W8, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A8 = tf.nn.relu(Z8)
    # CONV2D: filters W9, stride 1, padding 'SAME'
    Z9 = tf.nn.conv2d(A8,W9, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A9 = tf.nn.relu(Z9)
    # CONV2D: filters W10, stride 1, padding 'SAME'
    Z10 = tf.nn.conv2d(A9,W10, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A10 = tf.nn.relu(Z10)
    # MAXPOOL: window 2x2, sride 2, padding 'VALID'
    P4 = tf.nn.max_pool(A10, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # CONV2D: filters W11, stride 1, padding 'SAME'
    Z11 = tf.nn.conv2d(P4,W11, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A11 = tf.nn.relu(Z11)
    # CONV2D: filters W12, stride 1, padding 'SAME'
    Z12 = tf.nn.conv2d(A11,W12, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A12 = tf.nn.relu(Z12)
    # CONV2D: filters W13, stride 1, padding 'SAME'
    Z13 = tf.nn.conv2d(A12,W13, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A13 = tf.nn.relu(Z13)
    # MAXPOOL: window 2x2, sride 2, padding 'VALID'
    P5 = tf.nn.max_pool(A13, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # FLATTEN
    P5 = tf.contrib.layers.flatten(P5)
    # FULLY-CONNECTED with non-linear activation function.
    A14 = tf.contrib.layers.fully_connected(P5, 4096)
    A15 = tf.contrib.layers.fully_connected(A14, 4096)    
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z16 = tf.contrib.layers.fully_connected(A15, 6, activation_fn=None)
    ### END CODE HERE ###

    return Z16


# In[9]:

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z8 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z16, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z8 = " + str(a))


# **Expected Output**:
# 
# <table> 
#     <td> 
#     Z3 =
#     </td>
#     <td>
#     [[-0.44670227 -1.57208765 -1.53049231 -2.31013036 -1.29104376  0.46852064] <br>
#  [-0.17601591 -1.57972014 -1.4737016  -2.61672091 -1.00810647  0.5747785 ]]
#     </td>
# </table>

# In[10]:

# GRADED FUNCTION: compute_cost 

def compute_cost(Z16, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = Z16, labels = Y) )
    ### END CODE HERE ###
    
    return cost


# In[11]:

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z16 = forward_propagation(X, parameters)
    cost = compute_cost(Z16, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
    print("cost = " + str(a))


# **Expected Output**: 
# 
# <table>
#     <td> 
#     cost =
#     </td> 
#     
#     <td> 
#     2.91034
#     </td> 
# </table>

# In[12]:

# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z16 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z16, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X , Y:minibatch_Y})
                ### END CODE HERE ###
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z16, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters


# Run the following cell to train your model for 100 epochs. Check if your cost after epoch 0 and 5 matches our output. If not, stop the cell and go back to your code!

# In[1]:

_, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs = 2)


# **Expected output**: although it may not match perfectly, your expected output should be close to ours and your cost value should decrease.
# 
# <table> 
# <tr>
#     <td> 
#     **Cost after epoch 0 =**
#     </td>
# 
#     <td> 
#       1.917929
#     </td> 
# </tr>
# <tr>
#     <td> 
#     **Cost after epoch 5 =**
#     </td>
# 
#     <td> 
#       1.506757
#     </td> 
# </tr>
# <tr>
#     <td> 
#     **Train Accuracy   =**
#     </td>
# 
#     <td> 
#       0.940741
#     </td> 
# </tr> 
# 
# <tr>
#     <td> 
#     **Test Accuracy   =**
#     </td>
# 
#     <td> 
#       0.783333
#     </td> 
# </tr> 
# </table>

# In[ ]:



