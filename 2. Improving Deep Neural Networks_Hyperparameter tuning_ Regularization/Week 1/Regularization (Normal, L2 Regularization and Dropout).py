#Using Regularization in Deep Learning Models

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
np.random.seed(0)

train_X, train_Y, test_X, test_Y = load_2d_dataset()

# 1- Non Regularized Model (This is the baseline model)
def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True,lambd=0,keep_prob=1):
	grads = {}
	costs = []
	m = X.shape[1]
	layers_dims = [X.shape[0],20,3,1]

	parameters = initialize_parameters(layers_dims)

	for i in range(num_iterations):

		#Forward Propagation
		if keep_prob==1:
			a3, cache = forward_propagation(X, parameters)
		elif keep_prob<1:
			a3, cache = forward_propagation_with_dropout(X,parameters, keep_prob)

		#Compute Cost
		if keep_prob==1 || lambd=0 :
			cost = compute_cost(a3,Y)
		else
			cost = compute_cost_with_regularization(a3, Y, lambd)

		#Backward propagation
		if lambd == 0 and keep_prob == 1:
			grads = backward_propagation(X, Y, cache)
		elif lambd!=0:
			grads = backward_propagation_with_regularization(X,Y,cache,lambd)
		elif keep_prob < 1:
			grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)

		#gradient descent
		parameters = update_parameters(parameters, grads,learning_rate)

		if i%100==0 and print_cost:
			print("Cost after iteration %i:%f"%(i,cost))
		if i%100==0:
			costs.append(cost)
	plt.plot(costs)
	plt.xlabel('iterations')
	plt.ylabel('costs')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	return parameters

# 2- L2 Regularization
def compute_cost_with_regularization(A3,Y,parameters,lambd):
	
	m = X.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	W3 = parameters["W3"]

	cross_entropy_cost = compute_cost(A3,X)

	L2_regularization_cost = lambd * (np.sum(np.square(W1))) + np.sum(np.square(W2)) + np.sum(np.square(W3)) / (2 * m)

	cost = cross_entropy_cost + L2_regularization_cost

	return cost 

def backward_propagation_with_regularization(X, Y, cache, lambd):
	m = X.shape[1]
	Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3 = cache

	dZ3 = A3 - Y
	
	dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd * W3) / m
	db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)

	dA2 = np.dot(W3.T, dZ3)
	dZ2 = np.multiply(dA2, np.int64(A2>0))
	dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd * W2) / m
	db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

	dA1 = np.dot(W2.T, dZ2)
	dZ1 = np.multiply(dA1, np.int64(A1>0))
	dW1 = 1./m * np.dot(dZ1, X.T) + (lambd * W1) / m
	db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

	gradients = {"dZ3":dZ3, "dW3":dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

#3- Dropout Regularization

def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b2 = parameters["b3"]

	Z1 = np.dot(W1, X) + b1
	A1 = relu(Z1)
	D1 = np.random.randn(A1.shape[0],A1.shape[1])
	D1 = D1 < keep_prob
	A1 = A1 * D1
	A1 = A1 / keep_prob

	Z2 = np.dot(W2, A1) + b2
	A2 = relu(Z2)
	D2 = np.random.randn(A2.shape[0],A2.shape[1])
	D2 = D2 < keep_prob
	A2 = A2 * D2
	A2 = A2 / keep_prob

	Z3 = np.dot(W3,A2) + b3
	A3 = sigmoid(Z3)

	cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
	return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2               
    dA2 = dA2 / keep_prob              
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1          
    dA1 = dA1 / keep_prob   
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

#4 - Conclusion
# Here are the results of the three models:
# model 	train accuracy 	test accuracy
# 3-layer NN without regularization 	95% 	91.5%
# 3-layer NN with L2-regularization 	94% 	93%
# 3-layer NN with dropout 			93% 	95% 