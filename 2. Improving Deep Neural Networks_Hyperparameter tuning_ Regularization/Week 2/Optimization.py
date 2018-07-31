import numpy as np 
import math 
import sklearn
import matplotlib.pyplot as plt 
import scipy.io 

def update_parameters_with_gd(parameters, grads, learning_rate):
	L = parameters / 2

	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] 
		parameters["b" + str(l+1)] = parameters["b" + sr(l+1)] - learning_rate * grads["db" + str(l+1)]
	return parameters

def random_mini_batch(X, Y, mini_batch_size=64, seed=0):
	np.random.seed(seed)
	m = X.shape[1]
	mini_batches = []

	permutation = list(np.random.permutation(m))
	shuffled_X = X[:,permutation]
	shuffled_Y = Y[:,permutation].reshape((1,m))

	num_complete_minibatches = math.floor(m/mini_batch_size)

	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k * mini_batch_size:(k+1)*mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k+1)*mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m%mini_batch_size !=0:
		end = m - mini_batch_size * math.floor(m / mini_batch_size)
		mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches

