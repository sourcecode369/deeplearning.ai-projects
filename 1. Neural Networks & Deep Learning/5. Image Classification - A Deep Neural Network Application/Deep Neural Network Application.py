import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline


#load dataset
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#Display a single picture
index = 10
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0, index]) + ". Its a " + classes[train_y[0,index]].decode("utf-8") + " picture")

#Shape of the dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

#reshape the training and test examples

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T 
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T 

#standardize the data
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

#Model Architecture

## 1. Two-layer neural network
# def initialize_parameters(n_x, n_h, n_y):
#     ...
#     return parameters 
# def linear_activation_forward(A_prev, W, b, activation):
#     ...
#     return A, cache
# def compute_cost(AL, Y):
#     ...
#     return cost
# def linear_activation_backward(dA, cache, activation):
#     ...
#     return dA_prev, dW, db
# def update_parameters(parameters, grads, learning_rate):
#     ...
#     return parameters
n_x = 12288
n_h = 7
n_y = 1
layer_dims = (n_x,n_h,n_y)

def two_layer_model(X,Y,layer_dims,learning_rate=0.0075, num_iterations=3000,print_cost=False):
	grads = {}
	costs = []
	m = X.shape[1]

	parameters = initialize_parameters(n_x,n_h,n_y)
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	for i in range(0,num_iterations):
		A1, cache1 = linear_activation_forward(X,W1,b1,"relu")
		A2, cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")

		cost = compute_cost(A2,Y)
		costs.append(cost)

		dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        grads["dW1"] = dW1
        grads["dW2"] = dW2
        grads["db1"] = db1
        grads["db2"] = db2

        parameters = update_parameters(parameters,grads,learning_rate)

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        if print_cost and i%100 == 0:
        	print(cost)
        if i%100 == 0:
        	costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('num_iterations')
    plt.title('learning_rate = ' + str(learning_rate))
    plt.show()
    return parameters

 #run the model
 parameters = two_layer_model(train_x,train_y,layer_dims =(n_x,n_h,n_y), num_iterations=2500, print_cost=True)

 #predictions
 predictions_train = predict(train_x, train_y, parameters)

 predictions_test = predict(test_x,test_y,parameters)


#2 L-layer Neural Network
# def initialize_parameters_deep(layers_dims):
#     ...
#     return parameters 
# def L_model_forward(X, parameters):
#     ...
#     return AL, caches
# def compute_cost(AL, Y):
#     ...
#     return cost
# def L_model_backward(AL, Y, caches):
#     ...
#     return grads
# def update_parameters(parameters, grads, learning_rate):
#     ...
#     return parameters

#constants
layer_dims = [12288,20,7,5,1] # 4-layer model

#L-layer model
def L_Layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations,print_cost):
	costs = []
	parameters = initialize_parameters_deep(layers_dims)

	for i in range(0,num_iterations):
		AL, caches = L_model_forward(X, parameters)
		cost = compute_cost(AL,Y)
		grads = L_model_backward(AL,Y, caches)
		parameters = update_parameters(parameters,grads, learning_rate)

		if print_cost and i%100 == 0:
			print("Cost after iteration %i:%f"%(i,cost))
		if i%100 == 0:
			costs.append(cost)
	plt.plt(costs)
	plt.ylabel('cost')
	plt.xlabel('iteration')
	plt.title("learning_rate = " + str(learning_rate))
	plt.show()

	return parameters

#run the model
parameters = L_Layer_model(train_x,train_y,layer_dims,num_iterations,learning_rate,print_cost)

pred_train = predict(train_x,train_y,parameters)
pred_test = predict(test_x,test_y,parameters)

#mismatch labels
print_mislabeled_images(classes,test_x,test_y,pred_test)