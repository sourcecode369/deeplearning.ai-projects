import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn
%matplotlib inline

train_X, train_Y, test_X, test_Y = load_dataset()

def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization='he'):
	grads = {}
	costs = []
	m = X.shape[1]
	layers_dims = {X.shape[0],10,5,1}

	if initialization=="zeros"
		parameters = initialize_parameters_zeros(layers_dims)
	elif initialization=="random":
		parameters = initialize_parameters_random(layers_dims)
	elif initialization=="he":  
		parameters = initialize_parameters_he(layers_dims)

	for i in range(0, num_iterations):
		a3, cache = forward_propagation(X, parameters)

		cost = compute_cost(a3,Y)

		grads = backward_propagation(X, Y,cache)

		parameters = update_parameters(parameters, grads, learning_rate)

		if print_cost and i % 1000 == 0:
			print("Cost after iteration %d:%f"%(i,cost))
		if i % 100 == 0:
			costs.append(cost)
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('num_iterations')
	plt.title("learning_rate =" + str(learning_rate))
	plt.show()

	return parameters

def initialize_parameters_zeros(layers_dims):
	parameters = []
	L = len(parameters)
	for l in range(1,L):
		parameters["W" + str(l)] = np.zeros(shape=(layers_dims[l],layers_dims[l-1]))
		parameters["b" + str(l)] = np.zeros(shape=(layers_dims[l],1))
	return parameters

def initialize_parameters_random(layers_dims):
	parameters = []
	L = len(parameters)
	for l in range(1,L):
		parameters["W" + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*10
		parameters["b" + str(l)] = np.random.randn(layers_dims[l],1)
	return parameters

def initialize_parameters_he(layers_dims):
	parameters = []
	L = len(layers_dims)
	for i in range(layers_dims):
		parameters["W" + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
		parameters["b" + str(l)] = np.random.randn(layers_dims[l],1)
	return parameters