'''The Code below wont run as it is a wrapper of hwo the shallow neural network will work.'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

'''
	The Neural Network Model

	1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
	2. Initialize the model's parameters
	3. Loop:
	    - Implement forward propagation
	    - Compute loss
	    - Implement backward propagation to get the gradients
	    - Update parameters (gradient descent)
'''

#1.1
#defining the neural network structure

'''
--- 	n_x -> size of the input layer
---		n_h -> size of the hidden layer
---		n_y -> size of the output layer 
'''

def  layer_size(X,Y):
	n_x = X.shape[0]
	n_h = 4
	n_y = Y.shape[0]
	return (n_x, n_h,n_y)

#1.2
#intialize the models parameters

def inititalize_parameters(n_x,n_h,n_y):
	"""
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing our parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros(shape=(n_h,1))
	W2 = np.random.randn(n_y,n_h)
	b2 = np.zeros(n_y,1)

	parmaters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
	return parmaters

#1.3
#the loop
'''
	The steps implemented are:

	    Retrieve each parameter from the dictionary "parameters" (which is the output of initialize_parameters()) by using parameters[".."].
	    Implement Forward Propagation. 
	    Compute Z[1],A[1],Z[2]Z[1],A[1],Z[2] and A[2]A[2] (the vector of all your predictions on all the examples in the training set).

		Values needed in the backpropagation are stored in "cache". The cache will be given as an input to the backpropagation function.
'''

def forward_propagation(X, parameters):
	W1 = parameters["W1"]
	b1 = parameters['b1']
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	Z1 = np.dot(W1,X) + b1
	A1 = sigmoid(A)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)

	cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}

	return A2, cache

'''
	computed A[2]A[2] (in the Python variable "A2"), which contains a[2](i)a[2](i) for every example, we can compute the cost function as follows:

	-->	J=−1m∑i=0m(y(i)log(a[2](i))+(1−y(i))log(1−a[2](i)))(13)
'''
def compute_cost(A2,Y,parameters):
	'''
	Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- true labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
  	'''
	m = Y.shape[1]

	logprobs = np.multiply(Y,np.log(A2)) + np.multiply(1-Y,np.log(1-A2))
	cost = -1/m * np.sum(logprobs)

	cost = np.squeeze(cost)
	return cost


'''' Implement the function backward_propagation().'''
# Summary of Gradient Descent - https://raw.githubusercontent.com/sourcecode369/Deep-Learning-Projects/baf48c293ea0c63e36da3d8d76b29492351a1947/1.%20Neural%20Networks%20%26%20Deep%20Learning/3.%20Planar%20Data%20Classification%20with%20Shallow%20Neural%20Network/images/grad_summary.png

def backward_propagation(parameters,cache,X,Y):
	# Implement the backward propagation using the instructions above.
    
 #    Arguments:
 #    parameters -- python dictionary containing our parameters 
 #    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
 #    X -- input data of shape (2, number of examples)
 #    Y -- "true" labels vector of shape (1, number of examples)
    
 #    Returns:
 #    grads -- python dictionary containing your gradients with respect to different parameters
    

    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1,2))
    dW1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m * np.sum(dZ1,axis=1,keepdims=True)

    grads = {"dW2":dW2,"db2":db2,"dW1":dW1,"db1":db1}

    return grads

 '''Implement the update rule. Use gradient descent. We have to use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).

 General gradient descent rule: θ=θ−α∂J∂θθ=θ−α∂J∂θ where αα is the learning rate and θθ represents a parameter.'''

 def update_parameters(parameters,grads,learning_rate=1.2):
 	W1 = parameters["W1"]
 	b1 = parameters["b1"]
 	W2 = parameters["W2"]
 	b2 = parameters["b2"]

 	dW1 = grads["dW1"]
 	dW2 = grads["dW2"]
 	db1 = grads["db1"]
 	db2 = grads["db2"]

 	W1 = W1 - learning_rate*dW1
 	b1 = b1 - learning_rate*db1
 	W2 = W2 - learning_rate*dW2
 	b2 = b2 - learning_rate*db2

 	parameters = {"W1":W1,"W2":W2,"b1":b1,"b2":b2}

 	return parameters

 #1.4
 #Integrating 1.1, 1.2 and 1.3 in nn_model() 
 '''
	Build your neural network model in nn_model().
	(The Previous Functions must be implemented in the right order)
 '''

 def nn_model(X,Y,n_h,num_iterations = 10000, print_cost = False):
 	"""
		Arguments:
		X -- dataset of shape (2, number of examples)
		Y -- labels of shape (1, number of examples)
		n_h -- size of the hidden layer
		num_iterations -- Number of iterations in gradient descent loop
		print_cost -- if True, print the cost every 1000 iterations

		Returns:
		parameters -- parameters learnt by the model. They can then be used to predict.
    """
    n_x = layer_size(X,Y)[0]
    n_y = layer_size(X,Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
    	A2, cache = forward_propagation(X,parameters)

    	cost = compute_cost(A2, Y, parameters)

    	grads = backward_propagation(parameters,cache,X,Y)

    	parameters = update_parameters(parameters,grads)

    	if print_cost == True and i%100 == 0:
    		print("Cost after iteration %i:%f"%(i,cost))
    return parameters

#1.5
#Predictions

'''
	Use this model to predict by building predict(). Use forward propagation to predict results.

	Reminder: predictions = yprediction=𝟙{activation > 0.5}={10if activation>0.5otherwiseyprediction=1{activation > 0.5}={1if activation>0.50otherwise

	As an example, if you would like to set the entries of a matrix X to 0 and 1 based on a threshold you would do: X_new = (X > threshold)

'''

def predict(parameters,X):
	"""
	    Using the learned parameters, predicts a class for each example in X
	    
	    Arguments:
	    parameters -- python dictionary containing your parameters 
	    X -- input data of size (n_x, m)
	    
	    Returns
	    predictions -- vector of predictions of our model (red: 0 / blue: 1)
	"""
	# Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.

	A2, cache = forward_propagation(X,parameters)
	predict = A2 > 0.5
	predict =  100 * np.abs(np.mean(predict))
	return predict

if __name__ == '__main__'
	
	#Build a model with a n_h-dimensional hidden layer
	parameters = nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=True)

	# Plot the decision boundary
	plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
	plt.title("Decision Boundary for hidden layer size " + str(4))

	#print accuracy
	predictions = predict(parameters, X)
	print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


	'''
	Tuning the hidden layer size

	plt.figure(figsize=(16, 32))
	hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
	for i, n_h in enumerate(hidden_layer_sizes):
	    plt.subplot(5, 2, i+1)
	    plt.title('Hidden Layer of size %d' % n_h)
	    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
	    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
	    predictions = predict(parameters, X)
	    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
	    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
	'''

	# Interpretation:

	 #    The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data.
	 #    The best hidden layer size seems to be around n_h = 5. Indeed, a value around here seems to fits the data well without also incurring noticable overfitting.
	 #    You will also learn later about regularization, which lets you use very large models (such as n_h = 50) without much overfitting.

	 '''
		Performance on other datasets
	 '''
		# Datasets
	# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

	# datasets = {"noisy_circles": noisy_circles,
	#             "noisy_moons": noisy_moons,
	#             "blobs": blobs,
	#             "gaussian_quantiles": gaussian_quantiles}

	# ### START CODE HERE ### (choose your dataset)
	# dataset = "noisy_moons"
	# ### END CODE HERE ###

	# X, Y = datasets[dataset]
	# X, Y = X.T, Y.reshape(1, Y.shape[0])

	# # make blobs binary
	# if dataset == "blobs":
	#     Y = Y%2

	# # Visualize the data
	# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral); 