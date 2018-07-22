#Dont try to run this code as it wont execute. 
#It is merely a wrapper of the original work which uses some more files to use fuctions that hasnt been implemented in here.


# Importing the Dependencies
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# To build your neural network, you will be implementing several "helper functions". 
# Each small helper function you will implement will have detailed instructions that will walk you through the necessary steps. 
# Here is an outline of this assignment, you will:

    # Initialize the parameters for a two-layer network and for an L-layer neural network.
    # Implement the forward propagation module 
    #     Complete the LINEAR part of a layer's forward propagation step (resulting in Z[l]).
    #     We give you the ACTIVATION function (relu/sigmoid).
    #     Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
    #     Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer LL). This gives you a new L_model_forward function.
    # Compute the loss.
    # Implement the backward propagation module (denoted in red in the figure below).
    #     Complete the LINEAR part of a layer's backward propagation step.
    #     We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward)
    #     Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
    #     Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
    # Finally update the parameters.

# Note that for every forward function, there is a corresponding backward function. 
#That is why at every step of your forward module you will be storing some values in a cache. 
#The cached values are useful for computing gradients. 
#In the backpropagation module you will then use the cache to calculate the gradients. 
#This assignment will show you exactly how to carry out each of these steps.

## 1. Initialization
	 # We will write two helper functions that will initialize the parameters for your model. 
	 # The first function will be used to initialize parameters for a two layer model. 
	 # The second one will generalize this initialization process to L layers.

# 1.1 2-Layer Neural Network
def initialize parameters(n_x,n_h,n_y):
	"""
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h,n_x)*0.01
    W2 = np.random.randn(n_y,n_h)*0.01
    b1 = np.zeros(n_h,1)
    b2 = np.zeros(n_y,1)

    parameters = {"W1":W1,"W2":W2,"b1":b1,"b2":b2}

    return parameters

# 1.2 L-Layer Neural Network
# The initialization for a deeper L-layer neural network is more complicated 
# because there are many more weight matrices and bias vectors. 
# When completing the initialize_parameters_deep,
# you should make sure that your dimensions match between each layer. 
# Recall that n[l] is the number of units in layer l. 
def initialize_parameters_deep(layer_dims):
	#Exercise: Implement initialization for an L-layer Neural Network.

#Instructions:

    # The model's structure is [LINEAR -> RELU] ×× (L-1) -> LINEAR -> SIGMOID. 
    # I.e., it has L−1L−1 layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
    # Use random initialization for the weight matrices. Use np.random.randn(shape) * 0.01.
    # Use zeros initialization for the biases. Use np.zeros(shape).
    # We will store n[l]n[l], the number of units in different layers, in a variable layer_dims. 
    # For example, the layer_dims for the "Planar Data classification model" from last week would have been [2,4,1]: 
    # There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. 
    # Thus means W1's shape was (4,2), b1 was (4,1), W2 was (1,4) and b2 was (1,1). Now you will generalize this to LL layers!
    # Here is the implementation for L=1L=1 (one layer neural network). 
    # It should inspire you to implement the general case (L-layer neural network).

    #   if L == 1:
    #       parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
    #       parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
    parameters = []
    L = len(layer_dims)
    for l in range(1, L):
    	parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
    	parameters["b" + str(l)] = np.zeros(layer_dims[l],1)
    return parameters


## 2. Forward Propagation Module

# 2.1 Linear Forward
# Now that you have initialized your parameters, you will do the forward propagation module. 
# You will start by implementing some basic functions that you will use later when implementing the model. 
# You will complete three functions in this order:

#     LINEAR
#     LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid.
#     [LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID (whole model)

# The linear forward module (vectorized over all the examples) computes the following equations:

# Z[l]=W[l]A[l−1]+b[l]

# where A[0]=X.

# Exercise: Build the linear part of forward propagation.

# Reminder: The mathematical representation of this unit is Z[l]=W[l]A[l−1]+b[l]Z[l]=W[l]A[l−1]+b[l]. 
# You may also find np.dot() useful. 
# If your dimensions don't match, printing W.shape may help.

def linear_forward(A,W,b):
	"""
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W,A) + b
    cache = (A,W,b)
    return Z,cache

# 2.2 Linear Activation Forward
# Will implement a function that does the LINEAR forward step followed by an ACTIVATION forward step.
# Exercise: Implement the forward propagation of the LINEAR->ACTIVATION layer. 
# Mathematical relation is: A[l]=g(Z[l])=g(W[l]A[l−1]+b[l])A[l]=g(Z[l])=g(W[l]A[l−1]+b[l]) where the activation "g" can be sigmoid() or relu(). 
# Use linear_forward() and the correct activation function.
def linear_activation_forward(A_prev,W,b,activation):
	"""
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
    	Z, linear_cache = linear_forward(A_prev,W,b)
    	A, activation_cache = sigmoid(Z)
    elif activation == "relu":
    	Z, linear_cache = linear_forward(A_prev,W,b)
    	A, activation_cache = relu(Z)
    cache = (linear_cache,activation_cache)
    return A, cache

#Note: In deep learning, the "[LINEAR->ACTIVATION]" computation is counted as a single layer in the neural network, not two layers. 

## 2.3 Deep Neural Network Forward Propagation
# For even more convenience when implementing the L-layer Neural Net, 
# you will need a function that replicates the previous one (linear_activation_forward with RELU) L−1 times, 
# then follows that with one linear_activation_forward with SIGMOID.


# Exercise: Implement the forward propagation of the L - layer Deep Neural Network model.

# Instruction: In the code below, the variable AL will denote A[L]=σ(Z[L])=σ(W[L]A[L−1]+b[L])A[L]=σ(Z[L])=σ(W[L]A[L−1]+b[L]). (This is sometimes also called Yhat, i.e., this is Ŷ Y^.)

# Tips:

#     Use the functions you had previously written
#     Use a for loop to replicate [LINEAR->RELU] (L-1) times
#     Don't forget to keep track of the caches in the "caches" list. To add a new value c to a list, you can use list.append(c).


def L_model_forward(X, parameters):
	"""
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    
    caches = [] 
    A = X    
    L = len(parameters) / 2    
    for l in range(1,L):    	
    	A_prev = A
    	A, cache = linear_activation_forward(A,parameters["W" + str(l)],parameters["b" + str(l) + activation="relu"])
    	caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = 'sigmoid')
    caches.append(cache)
    return AL, caches


## Now you have a full forward propagation that takes the input X and 
## outputs a row vector A[L] containing your predictions. 
## It also records all intermediate values in "caches". 
## Using A[L], you can compute the cost of your predictions.

## 3 - The Cost Function
# Now we will implement forward and backward propagation. 
# We need to compute the cost, because We want to check if your model is actually learning.

def compute_cost(AL,Y):
	"""
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = (-1/m)*np.sum(np.multiply(Y,np.log(AL))+ np.multiply(1-Y,np.log(1-AL)))
    cost = np.squeeze(cost)
    return cost

## 4 - Backpropagation

# 4.1 - Linear Backward
# The three outputs (dW[l],db[l],dA[l])(dW[l],db[l],dA[l]) are computed using the input dZ[l]dZ[l].Here are the formulas you need:
# dW[l]=∂L∂W[l]=1mdZ[l]A[l−1]T
# db[l]=∂L∂b[l]=1m∑i=1mdZ[l](i)
# dA[l−1]=∂∂A[l−1]=W[l]TdZ[l]

def linear_backward(dZ, cache):
	"""
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = 1/m*np.multiply(dZ,cache[0].T)
	db = 1/m*np.sum(dZ,axis=1,keepdims=True)
	dA_prev = np.multiply(cache[1].T,dZ)

	return dA_prev, dW, db

# # 4.2 - Linear Activation Backward
# Next, you will create a function that merges the two helper functions:
# linear_backward and the backward step for the activation linear_activation_backward.

# To help you implement linear_activation_backward, we provided two backward functions:

#     sigmoid_backward: Implements the backward propagation for SIGMOID unit. You can call it as follows:

# dZ = sigmoid_backward(dA, activation_cache)

#     relu_backward: Implements the backward propagation for RELU unit. You can call it as follows:

# dZ = relu_backward(dA, activation_cache)

# If g(.)g(.) is the activation function, sigmoid_backward and relu_backward compute
# dZ[l]=dA[l]∗g′(Z[l])(11)
# dZ[l]=dA[l]∗g′(Z[l])
## Exercise: Implement the backpropagation for the LINEAR->ACTIVATION layer.
def linear_activation_backward(dA,cache,activation):
	"""
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
    	dZ = relu_backward(dA, activation_cache)
    	dA_prev, dW, db = linear_backward(dZ,linear_cache)
    elif activation == "sigmoid":
    	dZ = sigmoid_backward(dA,activation_cache)
    	dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev,dW,db

# # 4.3 Implement backpropagation for the [LINEAR->RELU] ×× (L-1) -> LINEAR -> SIGMOID model.
def L_model_backward(AL, Y, cache):
	"""
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - np.divide(Y,AL) - np.divide(1-Y,1-AL)
    grads["dA" + str(L-1)], grads["dW" + str(L-1)], grads["db" + str(L-1)] = linear_backward(sigmoid_backward(dAL,current_cache[1]),current_cache[0])
    for l in reversed(range(L-1)):
    	current_cache = caches[1]
    	dA_prev_temp, dW_temp,db_temp = linear_backward(sigmoid_backward(dAL,current_cache[1]),current_cache[0])
    	grads["dA" + str(l)] = dA_prev_temp
    	grads["dW" + str(l+1)] = dW_temp
    	grads["db" + str(l+1)] = db_temp
    return grads

# # 4.4 Update Parameters
# In this section you will update the parameters of the model, using gradient descent:

# W[l]=W[l]−α dW[l]
# b[l]=b[l]−α db[l]

# where α is the learning rate. After computing the updated parameters, store them in the parameters dictionary. 
def update_parameters(parameters, grads, learning_rate):
	"""
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) / 2
    for l in range(L):
    	parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
    	parameters["b" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
    return parameters

# This is most of the code required for building a L Layererd Deep Neural Network. 
# In the next section we will be implementing a Image Classification,
# Using the functions from this python file.  