import numpy as np 
import math 
import sklearn
import matplotlib.pyplot as plt 
import scipy.io 
#Gradient Descent 
def update_parameters_with_gd(parameters, grads, learning_rate):
	L = parameters / 2

	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] 
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
	return parameters
#Stochastic Gradient Descent
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

#Gradient Descent with Momentum
def initialize_velocity(parameters):
	L = len(parameters) / 2
	v = {}
	for l in range(L):
		v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
		v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
	return v
def gd_with_momentum(parameters,grads,v,beta,learning_rate):
	L = parameters/2
	for l in range(L):
		v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * v["dW" + str(l+1)]
		v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * v["db" + str(l+1)]

		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]

	return parameters,v

#Adam Optimizer
def adam_init(parameters):
	L = parameters/2
	v = {}
	s = {}
	for l in range(L):
		v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
		v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

		s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
		s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])

	return v, s
def adam_optimizer(parameters, grads, learning_rate, v,s,t, beta1, beta2, epsilon=1e-8):
	L = len(parameters) / 2
	v_corrected = {}
	s_corrected = {}

	for l in range(L):
		v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1)*grads["dW" + str(l+1)]
		v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1)*grads["db" + str(l+1)]

		s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * grads["dW" + str(l+1)]
		s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * grads["db" + str(l+1)]

		v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - (beta1) ** t)
		v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - (beta1) ** t)

		s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - (beta2) ** t)
		s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - (beta2) ** t)

		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)] / np.sqrt(s_corrected["dW" + str(l+1)] + epsilon))
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)] / np.sqrt(s_corrected["db" + str(l+1)] + epsilon))
		return parameters, s, v 	